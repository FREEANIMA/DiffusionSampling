(() => {
  const $ = (id) => document.getElementById(id);
  const modeEl = $('mode');

  // ---------- 자동완성 ----------
  const TAGS = window.TAG_LIST || [];
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
    String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

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
    if (out.length === 0 && q.length >= 3) {
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

  // ---------- Tag history ----------
  const histEl = $('tag_history');
  let histItems = [];

  const renderHistory = () => {
    if (!histItems.length) {
      histEl.innerHTML = '<span class="history-empty">기록 없음</span>';
      return;
    }
    histEl.innerHTML = histItems.map((it, i) =>
      `<button type="button" class="history-item" data-i="${i}" title="${escapeHtml(it.input)}">
         <span class="h-text">${escapeHtml(it.input || '(빈 입력)')}</span>
         <span class="h-count">+${it.tags.length}</span>
       </button>`
    ).join('');
  };

  const loadHistory = async () => {
    try {
      const r = await fetch('api/tag_history');
      const { items } = await r.json();
      histItems = items.slice().reverse();   // 최신순
      renderHistory();
    } catch (_) {}
  };

  histEl.addEventListener('click', (e) => {
    const btn = e.target.closest('.history-item');
    if (!btn) return;
    const it = histItems[+btn.dataset.i];
    if (!it) return;
    const base = it.input.split(',').map((s) => s.trim()).filter(Boolean);
    tagsEl.value = [...new Set([...base, ...it.tags])].join(', ');
    $('fixed').checked = true;   // 확장된 태그를 그대로 고정해서 사용
  });

  $('history_refresh').onclick = loadHistory;
  loadHistory();

  // ---------- Mask / Rect ----------
  let localMask = null;   // inpainting: brush 마스크 (dataURL)
  let rect = null;        // detailer: {x, y, w, h} (원본 이미지 픽셀 좌표)
  let rectStart = null;
  let sourceImage = null;
  let drawing = false;

  const isDetailer = () => modeEl.value === 'detailer';

  $('load_image').onclick = async () => {
    try {
      const response = await fetch('api/load_image', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: $('img_path').value }),
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
        $('mask-status').textContent = isDetailer()
          ? '이미지 로드 완료. 드래그로 영역을 선택하세요.'
          : '이미지 로드 완료.';
        localMask = null;
        rect = null;
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
      (e.clientY - r.top) * canvas.height / r.height,
    ];
  };

  // inpainting: brush
  const draw = (e) => {
    const [x, y] = getXY(e);
    ctx.lineWidth = Number($('brush').value) || 20;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = 'rgba(255,0,0,0.6)';
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();   // 같은 path 재-stroke 방지 (알파 누적 완화)
    ctx.moveTo(x, y);
  };

  // detailer: 정사각형 (드래그 방향 기준, 이미지 경계 안으로 제한)
  const drawRect = (a, b) => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    const sx = dx >= 0 ? 1 : -1;
    const sy = dy >= 0 ? 1 : -1;

    // 드래그 방향으로 남은 공간
    const maxSide = Math.min(
      sx > 0 ? canvas.width - a[0] : a[0],
      sy > 0 ? canvas.height - a[1] : a[1],
    );
    const side = Math.max(0, Math.min(Math.max(Math.abs(dx), Math.abs(dy)), maxSide));

    const x = sx > 0 ? a[0] : a[0] - side;
    const y = sy > 0 ? a[1] : a[1] - side;

    ctx.fillStyle = 'rgba(255,0,0,0.25)';
    ctx.fillRect(x, y, side, side);
    ctx.lineWidth = Math.max(2, canvas.width / 400);
    ctx.strokeStyle = 'rgba(255,0,0,0.95)';
    ctx.strokeRect(x, y, side, side);
    return { x, y, w: side, h: side };
  };

  canvas.addEventListener('pointerdown', (e) => {
    if (!sourceImage) return;

    drawing = true;
    canvas.setPointerCapture(e.pointerId);
    localMask = null;
    rect = null;

    if (isDetailer()) {
      rectStart = getXY(e);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      $('mask-status').textContent = '영역 선택 중...';
      return;
    }

    const [x, y] = getXY(e);
    ctx.beginPath();
    ctx.moveTo(x, y);
    draw(e);
    $('mask-status').textContent = '마스크 수정 중...';
  });

  canvas.addEventListener('pointermove', (e) => {
    if (!drawing) return;
    if (isDetailer()) drawRect(rectStart, getXY(e));
    else draw(e);
  });

  const endDraw = (e) => {
    if (!drawing) return;
    drawing = false;

    if (isDetailer() && rectStart) {
      const r = drawRect(rectStart, getXY(e));
      if (r.w >= 8 && r.h >= 8) {
        rect = {
          x: Math.round(r.x), y: Math.round(r.y),
          w: Math.round(r.w), h: Math.round(r.h),
        };
        $('mask-status').textContent = `영역 선택 완료 (${rect.w}×${rect.h})`;
      } else {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        $('mask-status').textContent = '영역이 너무 작아요. 다시 드래그하세요.';
      }
      rectStart = null;
    }
  };

  canvas.addEventListener('pointerup', endDraw);
  canvas.addEventListener('pointercancel', () => { drawing = false; rectStart = null; });

  $('mask-clear').onclick = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    localMask = null;
    rect = null;
    $('mask-status').textContent = '초기화 완료.';
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

  // ---------- UI ----------
  const START_T_DEFAULT = { inpainting: '0.9', detailer: '0.85' };

  const applyMode = () => {
    const m = modeEl.value;
    const isT2I = m === 'txt2img';
    const isDet = m === 'detailer';

    $('txt2img-box').style.display = isT2I ? 'block' : 'none';
    $('inpaint-box').style.display = isT2I ? 'none' : 'block';
    $('inpaint-title').textContent = isDet ? 'Detailer' : 'Inpainting';
    $('brush-field').style.display = isDet ? 'none' : 'block';
    $('mask-finish').style.display = isDet ? 'none' : 'inline-block';
    $('detailer-box').style.display = isDet ? 'block' : 'none';
    if (START_T_DEFAULT[m]) $('start_t').value = START_T_DEFAULT[m];
  };

  modeEl.onchange = () => {
    applyMode();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    localMask = null;
    rect = null;
    $('mask-status').textContent = isDetailer()
      ? '드래그로 영역을 선택하세요.'
      : '마스크를 그린 뒤 Finish mask를 눌러주세요.';
  };
  applyMode();

  $('clear_out').onclick = () => {
    $('out').textContent = '';
    $('console').textContent = '';
    const result = $('result');
    if (result) result.innerHTML = '';
  };

  // ---------- Progress ----------
  let pollTimer = null;

  const startPoll = () => {
    const bar = $('bar');
    bar.style.display = 'block';
    bar.value = 0;
    bar.max = 1;

    pollTimer = setInterval(async () => {
      try {
        const r = await fetch('api/progress');
        const p = await r.json();
        if (!pollTimer) return;

        if (p.total > 0) {
          bar.max = p.total;
          bar.value = p.step;
          $('out').textContent = p.step >= p.total
            ? 'decoding / saving...'
            : `sampling ${p.step}/${p.total}`;
        } else {
          $('out').textContent = 'preparing...';
        }
      } catch (_) {}
    }, 300);
  };

  const stopPoll = () => {
    clearInterval(pollTimer);
    pollTimer = null;
    $('bar').style.display = 'none';
  };

  // ---------- Console ----------
  const consoleEl = $('console');

  setInterval(async () => {
    try {
      const r = await fetch('api/logs');
      const { lines } = await r.json();
      if (!lines.length) return;
      const stick = consoleEl.scrollTop + consoleEl.clientHeight >= consoleEl.scrollHeight - 5;
      consoleEl.textContent += lines.join('\n') + '\n';
      if (stick) consoleEl.scrollTop = consoleEl.scrollHeight;
    } catch (_) {}
  }, 500);

  // ---------- Generate ----------
  $('run').onclick = async () => {
    const params = {
      mode: modeEl.value,
      tags: $('tags').value,
      fixed: $('fixed').checked,
      nsfw: $('nsfw').value === 'true',
      sampler: $('sampler').value,
      cfg: $('cfg').value,
      steps: $('steps').value,
      seed: $('seed').value,
      H: $('H').value,
      W: $('W').value,
      img_path: $('img_path').value,
      start_t: $('start_t').value,
      neg_tags: $('neg_tags').value,
      n_cfg: $('n_cfg').value,
      mask: localMask,
      rect: rect,
    };

    if (params.mode === 'inpainting' && !localMask) {
      $('out').textContent = '마스크를 그린 뒤 Finish Mask를 눌러주세요.';
      return;
    }
    if (params.mode === 'detailer' && !rect) {
      $('out').textContent = '드래그로 사각형 영역을 선택해 주세요.';
      return;
    }

    const runBtn = $('run');
    runBtn.disabled = true;
    $('out').textContent = 'running...';
    startPoll();

    let result = $('result');
    if (!result) {
      result = document.createElement('div');
      result.id = 'result';
      $('out').after(result);
    }
    result.innerHTML = '';

    try {
      const response = await fetch('api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.error || 'Generation failed');
      }

      const img = document.createElement('img');
      img.src = `data:image/png;base64,${data.image}`;
      img.style.maxWidth = '512px';
      img.style.width = '100%';
      img.style.border = '1px solid #ccc';
      img.style.marginTop = '8px';

      result.append(img);
      $('out').textContent = 'done';

    } catch (e) {
      $('out').textContent = `failed: ${e.message}`;

    } finally {
      stopPoll();
      runBtn.disabled = false;
      loadHistory();
    }
  };

})();