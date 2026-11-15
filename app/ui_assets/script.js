
(() => {
  const $ = (id) => document.getElementById(id);
  const modeEl = $('mode');

  // ---------- 자동완성 ----------
  const TAGS = window.TAG_LIST || [];
  // 비교용 키: 소문자 + 공백/언더스코어 통일
  const key = (s) => s.trim().toLowerCase().replace(/[\s_]+/g, ' ');
  const KEYED = TAGS.map((t) => [t, key(t)]);

  const acEl = $('ac');
  const tagsEl = $('tags');
  let acItems = [], acIdx = -1;

  const lev = (a, b) => {
    const m = a.length, n = b.length;
    if (Math.abs(m - n) > 2) return 99;
    let prev = Array.from({ length: n + 1 }, (_, j) => j);
    for (let i = 1; i <= m; i++) {
      const cur = [i];
      for (let j = 1; j <= n; j++) {
        cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1,
                          prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
      }
      prev = cur;
    }
    return prev[n];
  };

  const currentToken = () => {
    const v = tagsEl.value;
    const pos = tagsEl.selectionStart ?? v.length;
    const start = v.lastIndexOf(',', pos - 1) + 1;
    return { start, end: pos, text: v.slice(start, pos) };
  };

  const escapeHtml = (s) =>
    String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  const search = (raw) => {
    const q = key(raw);
    if (!q) return [];
    const pre = [], sub = [];
    for (const [t, k] of KEYED) {
      if (k.startsWith(q)) pre.push(t);
      else if (k.includes(q)) sub.push(t);
      if (pre.length >= 8) break;
    }
    let out = pre.concat(sub).slice(0, 8);
    if (out.length === 0 && q.length >= 3) {      // 후보 없으면 오타 교정
      out = KEYED.map(([t, k]) => [t, lev(q, k)])
                 .filter(([, d]) => d <= 2)
                 .sort((a, b) => a[1] - b[1])
                 .slice(0, 8).map(([t]) => t);
    }
    return out;
  };

  const closeAc = () => { acEl.style.display = 'none'; acIdx = -1; };
  const renderAc = () => {
    if (!acItems.length) return closeAc();
    acEl.innerHTML = acItems.map((t, i) =>
      `<div class="ac-item${i === acIdx ? ' on' : ''}" data-i="${i}">${escapeHtml(t)}</div>`
    ).join('');
    acEl.style.display = 'block';
  };

  const pick = (tag) => {
    const { start, end } = currentToken();
    const v = tagsEl.value;
    const before = v.slice(0, start).replace(/\s+$/, '');
    const sep = before ? (before.endsWith(',') ? ' ' : ', ') : '';
    const after = v.slice(end).replace(/^\s*,?\s*/, '');
    tagsEl.value = before + sep + tag + ', ' + after;
    closeAc();
    tagsEl.focus();
  };

  tagsEl.addEventListener('input', () => {
    acItems = search(currentToken().text);
    acIdx = -1;
    renderAc();
  });
  tagsEl.addEventListener('keydown', (e) => {
    if (acEl.style.display === 'none') return;
    if (e.key === 'ArrowDown') { acIdx = (acIdx + 1) % acItems.length; renderAc(); e.preventDefault(); }
    else if (e.key === 'ArrowUp') { acIdx = (acIdx - 1 + acItems.length) % acItems.length; renderAc(); e.preventDefault(); }
    else if ((e.key === 'Enter' || e.key === 'Tab') && acIdx >= 0) { pick(acItems[acIdx]); e.preventDefault(); }
    else if (e.key === 'Escape') closeAc();
  });
  acEl.addEventListener('mousedown', (e) => {
    const i = e.target.dataset.i;
    if (i !== undefined) { pick(acItems[+i]); e.preventDefault(); }
  });
  tagsEl.addEventListener('blur', () => setTimeout(closeAc, 100));

  // ---------- Local Mask ----------
  let localMask = null;
  let sourceImage = null;
  let drawing = false;

  if (window.LOCAL_MODE) {
      $('load_image').onclick = async () => {
          try {
              const response = await fetch('/api/load_image', {
                  method: 'POST',
                  headers: {'Content-Type': 'application/json'},
                  body: JSON.stringify({path: $('img_path').value})
              });

              const data = await response.json();
              if (!response.ok || !data.success) throw new Error(data.error);

              sourceImage = new Image();
              sourceImage.onload = () => {
                  const bg = $('bg-canvas');
                  const mask = $('mask-canvas');

                  bg.width = mask.width = sourceImage.width;
                  bg.height = mask.height = sourceImage.height;

                  bg.getContext('2d').drawImage(sourceImage, 0, 0);
                  mask.getContext('2d').clearRect(0, 0, mask.width, mask.height);

                  $('mask-editor').style.display = 'block';
                  $('mask-status').textContent = '이미지 로드 완료.';
                  localMask = null;
              };

              sourceImage.onerror = () => {
                  $('mask-status').textContent = '이미지 로드 실패.';
              };

              sourceImage.src = `data:image/png;base64,${data.image}`;

          } catch (e) {
              $('out').textContent = `failed: ${e.message}`;
          }
      };

      const canvas = $('mask-canvas');
      const ctx = canvas.getContext('2d');

      const getXY = (e) => {
          const r = canvas.getBoundingClientRect();
          return [
              (e.clientX - r.left) * canvas.width / r.width,
              (e.clientY - r.top) * canvas.height / r.height
          ];
      };

      const draw = (e) => {
          const [x, y] = getXY(e);
          ctx.lineWidth = Number($('brush').value) || 20;
          ctx.lineCap = 'round';
          ctx.lineJoin = 'round';
          ctx.strokeStyle = 'rgba(255,0,0,0.6)';
          ctx.lineTo(x, y);
          ctx.stroke();
      };

      canvas.addEventListener('pointerdown', (e) => {
          if (!sourceImage) return;

          drawing = true;
          canvas.setPointerCapture(e.pointerId);

          const [x, y] = getXY(e);
          ctx.beginPath();
          ctx.moveTo(x, y);
          draw(e);

          localMask = null;
          $('mask-status').textContent = '마스크 수정 중...';
      });

      canvas.addEventListener('pointermove', (e) => {
          if (drawing) draw(e);
      });

      canvas.addEventListener('pointerup', () => drawing = false);
      canvas.addEventListener('pointercancel', () => drawing = false);

      $('mask-clear').onclick = () => {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          localMask = null;
          $('mask-status').textContent = '마스크 초기화 완료.';
      };

      $('mask-finish').onclick = () => {
          const output = document.createElement('canvas');
          output.width = canvas.width;
          output.height = canvas.height;

          const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height);
          const outCtx = output.getContext('2d');
          const outPixels = outCtx.createImageData(canvas.width, canvas.height);

          for (let i = 0; i < pixels.data.length; i += 4) {
              const value = pixels.data[i + 3] > 30 ? 255 : 0;
              outPixels.data[i] = value;
              outPixels.data[i + 1] = value;
              outPixels.data[i + 2] = value;
              outPixels.data[i + 3] = 255;
          }

          outCtx.putImageData(outPixels, 0, 0);
          localMask = output.toDataURL('image/png');

          $('mask-status').textContent = '마스크 생성 완료.';
      };
  }

  // ---------- 기존 UI ----------
  modeEl.onchange = () => {
    const isT2I = modeEl.value === 'txt2img';
    $('txt2img-box').style.display = isT2I ? 'block' : 'none';
    $('inpaint-box').style.display = isT2I ? 'none' : 'block';
  };


  $('clear_out').onclick = () => {
      $('out').textContent = '';

      const result = $('result');
      if (result) result.innerHTML = '';
  };

  $('run').onclick = async () => {
      const params = {
          mode: modeEl.value,
          tags: $('tags').value,
          nsfw: $('nsfw').value === 'true',
          sampler: $('sampler').value,
          cfg: $('cfg').value,
          steps: $('steps').value,
          seed: $('seed').value,
          H: $('H').value,
          W: $('W').value,
          img_path: $('img_path').value,
          start_t: $('start_t').value,
          mask: localMask,
      };

      if (window.LOCAL_MODE && params.mode === 'inpainting' && !localMask) {
          $('out').textContent = '마스크를 그린 뒤 Finish Mask를 눌러주세요.';
          return;
      }

      // Colab 방식 유지
      if (!window.LOCAL_MODE) {
          window.ui_params = params;
          $('out').textContent = 'running...';
          return;
      }

      const runBtn = $('run');
      runBtn.disabled = true;
      $('out').textContent = 'running...';

      let result = $('result');

      if (!result) {
          result = document.createElement('div');
          result.id = 'result';
          $('out').after(result);
      }

      result.innerHTML = '';

      try {
          const response = await fetch('/api/generate', {
              method: 'POST',
              headers: {
                  'Content-Type': 'application/json',
              },
              body: JSON.stringify(params),
          });

          const data = await response.json();

          if (!response.ok || !data.success) {
              throw new Error(data.error || 'Generation failed');
          }

          const path = document.createElement('div');
          path.textContent = `Saved: ${data.path}`;

          const img = document.createElement('img');
          img.src = `data:image/png;base64,${data.image}`;
          img.style.maxWidth = '512px';
          img.style.width = '100%';
          img.style.border = '1px solid #ccc';
          img.style.marginTop = '8px';

          result.append(path, img);

          $('out').textContent = 'done';

      } catch (e) {
          $('out').textContent = `failed: ${e.message}`;

      } finally {
          runBtn.disabled = false;
      }
  };

})();
