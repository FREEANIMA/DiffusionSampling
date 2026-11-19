(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);

  // 모드 → 요소 id 접두사
  const PREFIX = {
    txt2img: 't2i',
    inpainting: 'inp',
    detailer: 'det',
  };

  const key = (s) => s.trim().toLowerCase().replace(/[\s_]+/g, ' ');

  const escapeHtml = (s) =>
    String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');

  // =========================================================
  // Tabs
  // =========================================================

  const tabBtns = [...document.querySelectorAll('.tab')];
  const panes = [...document.querySelectorAll('.pane')];

  let activeMode = 'txt2img';

  const setTab = (mode) => {
    if (!PREFIX[mode]) mode = 'txt2img';
    activeMode = mode;

    tabBtns.forEach((b) => {
      const on = b.dataset.tab === mode;
      b.classList.toggle('active', on);
      b.setAttribute('aria-selected', String(on));
      b.tabIndex = on ? 0 : -1;
    });

    panes.forEach((p) => {
      p.hidden = p.dataset.mode !== mode;
    });

    // 숨겨져 있던 탭의 태그 박스는 scrollHeight가 0이었으므로 보인 뒤에 높이 재계산
    requestAnimationFrame(() => {
      document.querySelectorAll('.tag-box').forEach((el) =>
        el.dispatchEvent(new Event('tagbox:resize'))
      );
    });

    try { localStorage.setItem('mmdit.tab', mode); } catch (_) { /* ignore */ }

    if (location.hash !== '#' + mode) {
      window.history.replaceState(null, '', '#' + mode);
    }
  };

  tabBtns.forEach((b, i) => {
    b.addEventListener('click', () => setTab(b.dataset.tab));

    // 좌우 방향키로 탭 이동
    b.addEventListener('keydown', (e) => {
      if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
      const next = tabBtns[(i + (e.key === 'ArrowRight' ? 1 : -1) + tabBtns.length) % tabBtns.length];
      next.focus();
      setTab(next.dataset.tab);
      e.preventDefault();
    });
  });

  window.addEventListener('hashchange', () => {
    const m = location.hash.slice(1);
    if (PREFIX[m] && m !== activeMode) setTab(m);
  });

  // =========================================================
  // Tag box (자동으로 늘어나는 textarea)
  // 다른 초기화보다 먼저 실행해서, 이후 코드가 .value 를 읽고 쓸 때
  // 높이가 항상 따라가도록 한다.
  // =========================================================

  (function initTagBoxes() {
    const MAX_H = 240;
    const valueDesc = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value');

    function autoGrow(el) {
      el.style.height = 'auto';
      el.style.height = Math.min(el.scrollHeight + 2, MAX_H) + 'px';
    }

    document.querySelectorAll('.tag-box').forEach((el) => {
      // 코드에서 el.value = ... 로 바꿀 때도 높이 갱신 (히스토리/태그 트리 클릭 등)
      Object.defineProperty(el, 'value', {
        get() { return valueDesc.get.call(this); },
        set(v) { valueDesc.set.call(this, v); autoGrow(this); },
      });

      el.addEventListener('input', () => {
        // 붙여넣기로 들어온 줄바꿈 제거
        if (el.value.includes('\n')) {
          const pos = el.selectionStart;
          el.value = el.value.replace(/\s*\n+\s*/g, ' ');
          el.setSelectionRange(pos, pos);
        }
        autoGrow(el);
      });

      // Enter로 줄바꿈 방지 (생성 단축키/자동완성 선택과 충돌 방지)
      el.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') e.preventDefault();
      });

      // 탭 전환 시 높이 재계산용 (input 이벤트를 쓰면 자동완성이 같이 열림)
      el.addEventListener('tagbox:resize', () => autoGrow(el));

      autoGrow(el);
    });
  })();

  // =========================================================
  // Autocomplete
  // =========================================================

  const TAGS = window.TAG_LIST || [];
  const KEYED = TAGS.map((t) => [t, key(t)]);

  const lev = (a, b) => {
    const m = a.length;
    const n = b.length;
    if (Math.abs(m - n) > 2) return 99;

    let prev = Array.from({ length: n + 1 }, (_, j) => j);

    for (let i = 1; i <= m; i++) {
      const cur = [i];
      for (let j = 1; j <= n; j++) {
        cur[j] = Math.min(
          prev[j] + 1,
          cur[j - 1] + 1,
          prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1)
        );
      }
      prev = cur;
    }
    return prev[n];
  };

  const searchTags = (raw) => {
    const q = key(raw);
    if (!q) return [];

    const pre = [];
    const sub = [];

    for (const [t, k] of KEYED) {
      if (k.startsWith(q)) pre.push(t);
      else if (k.includes(q)) sub.push(t);
      if (pre.length >= 8) break;
    }

    let out = pre.concat(sub).slice(0, 8);

    if (out.length === 0 && q.length >= 3) {
      out = KEYED
        .map(([t, k]) => [t, lev(q, k)])
        .filter(([, d]) => d <= 2)
        .sort((a, b) => a[1] - b[1])
        .slice(0, 8)
        .map(([t]) => t);
    }
    return out;
  };

  const initAutocomplete = (tagsEl, acEl) => {
    let items = [];
    let idx = -1;

    const currentToken = () => {
      const v = tagsEl.value;
      const pos = tagsEl.selectionStart ?? v.length;
      const start = v.lastIndexOf(',', pos - 1) + 1;
      return { start, end: pos, text: v.slice(start, pos) };
    };

    const close = () => {
      acEl.style.display = 'none';
      idx = -1;
    };

    const render = () => {
      if (!items.length) return close();

      acEl.innerHTML = items.map((t, i) =>
        `<div class="ac-item${i === idx ? ' on' : ''}" data-i="${i}">${escapeHtml(t)}</div>`
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
      close();
      tagsEl.focus();
    };

    tagsEl.addEventListener('input', () => {
      items = searchTags(currentToken().text);
      idx = -1;
      render();
    });

    tagsEl.addEventListener('keydown', (e) => {
      if (acEl.style.display === 'none') return;

      if (e.key === 'ArrowDown') {
        idx = (idx + 1) % items.length;
        render();
        e.preventDefault();
      } else if (e.key === 'ArrowUp') {
        idx = (idx - 1 + items.length) % items.length;
        render();
        e.preventDefault();
      } else if ((e.key === 'Enter' || e.key === 'Tab') && idx >= 0) {
        pick(items[idx]);
        e.preventDefault();
      } else if (e.key === 'Escape') {
        close();
      }
    });

    acEl.addEventListener('mousedown', (e) => {
      const i = e.target.dataset.i;
      if (i !== undefined) {
        pick(items[+i]);
        e.preventDefault();
      }
    });

    tagsEl.addEventListener('blur', () => setTimeout(close, 100));
  };

  // =========================================================
  // Tag tools (Autocomplete + History + Tree) — 탭별 인스턴스
  // =========================================================

  // 트리 데이터는 한 번만 받아서 탭들이 공유
  let treePromise = null;
  const getTree = () => {
    if (!treePromise) {
      treePromise = fetch('api/tag_tree')
        .then((r) => {
          if (!r.ok) throw new Error('Tree request failed');
          return r.json();
        })
        .then((d) => d.tree || {})
        .catch((e) => {
          treePromise = null;   // 다음에 다시 시도할 수 있게
          throw e;
        });
    }
    return treePromise;
  };

  const TREE_LIMIT = 200;

  const initTagTools = (p) => {
    const tagsEl = $(`${p}-tags`);
    initAutocomplete(tagsEl, $(`${p}-ac`));

    // ----- History -----
    const histEl = $(`${p}-tag_history`);
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
        if (!r.ok) throw new Error('History request failed');
        const data = await r.json();
        histItems = (data.items || []).slice().reverse();
        renderHistory();
      } catch (_) { /* ignore */ }
    };

    histEl.addEventListener('click', (e) => {
      const btn = e.target.closest('.history-item');
      if (!btn) return;

      const it = histItems[+btn.dataset.i];
      if (!it) return;

      const base = it.input.split(',').map((s) => s.trim()).filter(Boolean);
      tagsEl.value = [...new Set([...base, ...it.tags])].join(', ');
      $(`${p}-fixed`).checked = true;
    });

    $(`${p}-history_refresh`).onclick = loadHistory;
    loadHistory();

    // ----- Tree -----
    const treeEl = $(`${p}-tag_tree`);
    const treeSearchEl = $(`${p}-tree_search`);
    let treeData = {};

    const addTag = (tag) => {
      const cur = tagsEl.value.split(',').map((s) => s.trim()).filter(Boolean);
      if (!cur.includes(tag)) cur.push(tag);
      tagsEl.value = cur.join(', ') + ', ';
      tagsEl.focus();
    };

    const fillGroup = (box, tags, q) => {
      const list = q ? tags.filter(([t]) => key(t).includes(q)) : tags;
      const shown = list.slice(0, TREE_LIMIT);

      box.innerHTML = shown.map(([t, c]) =>
        `<button type="button" class="tree-tag" data-tag="${escapeHtml(t)}">
          ${escapeHtml(t)}<span class="tt-count">${c}</span>
        </button>`
      ).join('') +
        (list.length > TREE_LIMIT
          ? `<span class="tree-more">+${list.length - TREE_LIMIT} more (검색으로 좁히기)</span>`
          : '');

      return list.length;
    };

    const renderTree = () => {
      const q = key(treeSearchEl.value);
      const types = Object.keys(treeData).sort();

      if (!types.length) {
        treeEl.innerHTML = '<span class="history-empty">태그 없음</span>';
        return;
      }

      treeEl.innerHTML = '';

      for (const typ of types) {
        const tags = treeData[typ];

        let matched = tags.length;
        if (q) {
          matched = tags.reduce((n, [t]) => n + (key(t).includes(q) ? 1 : 0), 0);
          if (!matched) continue;
        }

        const d = document.createElement('details');
        d.className = 'tree-group';
        d.dataset.type = typ;
        d.innerHTML =
          `<summary>
            <span class="t-name">${escapeHtml(typ)}</span>
            <span class="h-count">${matched}</span>
          </summary>
          <div class="tree-items"></div>`;

        const box = d.querySelector('.tree-items');

        if (q) {
          fillGroup(box, tags, q);
          d.open = true;
        } else {
          // 펼칠 때 처음 한 번만 렌더 (수만 개 DOM 방지)
          d.addEventListener('toggle', () => {
            if (d.open && !box.dataset.done) {
              fillGroup(box, tags, '');
              box.dataset.done = '1';
            }
          });
        }

        treeEl.appendChild(d);
      }

      if (!treeEl.children.length) {
        treeEl.innerHTML = '<span class="history-empty">검색 결과 없음</span>';
      }
    };

    let treeTimer = null;
    treeSearchEl.addEventListener('input', () => {
      clearTimeout(treeTimer);
      treeTimer = setTimeout(renderTree, 150);
    });

    treeEl.addEventListener('click', (e) => {
      const btn = e.target.closest('.tree-tag');
      if (btn) addTag(btn.dataset.tag);
    });

    getTree()
      .then((t) => { treeData = t; renderTree(); })
      .catch(() => {
        treeEl.innerHTML = '<span class="history-empty">불러오기 실패</span>';
      });

    return { loadHistory };
  };

  // =========================================================
  // Image editor (inpainting 마스크 / detailer 사각형) — 탭별 인스턴스
  // kind: 'mask' | 'rect'
  // =========================================================

  // ui.py 가 window.DETAILER_CFG 로 내려줌 (없으면 기본값)
  const CFG = window.DETAILER_CFG || {};
  const UNIT = CFG.unit || 32;          // patch_size * compression_ratio
  const ALLOW = CFG.allowable || 1024;  // allowable_size

  const initEditor = (p, kind) => {
    const isRect = kind === 'rect';

    const canvas = $(`${p}-mask-canvas`);
    const bg = $(`${p}-bg-canvas`);
    const ctx = canvas.getContext('2d');
    const statusEl = $(`${p}-mask-status`);
    const outEl = $(`${p}-out`);

    let sourceImage = null;
    let drawing = false;
    let localMask = null;
    let rect = null;
    let rectStart = null;

    const setStatus = (t) => { statusEl.textContent = t; };

    // ----- Crop 확대 미리보기 + 마스크 (detailer) -----
    const cropCanvas = isRect ? $(`${p}-crop-canvas`) : null;
    const cropMask = isRect ? $(`${p}-crop-mask-canvas`) : null;
    const cropStage = isRect ? $(`${p}-crop-stage`) : null;
    const cropEmpty = isRect ? $(`${p}-crop-empty`) : null;
    const cropInfo = isRect ? $(`${p}-crop-info`) : null;
    const cropMaskCtx = isRect ? cropMask.getContext('2d') : null;

    const showCrop = (r) => {
      if (!cropCanvas) return;

      if (!r || r.w <= 0 || r.h <= 0 || !sourceImage) {
        cropStage.style.display = 'none';
        cropEmpty.style.display = '';
        cropInfo.textContent = '';
        return;
      }

      cropCanvas.width = cropMask.width = r.w;
      cropCanvas.height = cropMask.height = r.h;
      cropCanvas.getContext('2d').drawImage(sourceImage, r.x, r.y, r.w, r.h, 0, 0, r.w, r.h);

      cropStage.style.display = 'block';
      cropEmpty.style.display = 'none';
      const k = ALLOW / Math.max(r.w, r.h);
      cropInfo.textContent = `${r.w}×${r.h} → ${Math.round(r.w * k)}×${Math.round(r.h * k)}`;
    };
    
    if (isRect) {
      const cropBrush = $(`${p}-brush`);
      let cropDrawing = false;

      const cropXY = (e) => {
        const b = cropMask.getBoundingClientRect();
        return [
          (e.clientX - b.left) * cropMask.width / b.width,
          (e.clientY - b.top) * cropMask.height / b.height,
        ];
      };

      const cropStroke = (e) => {
        const [x, y] = cropXY(e);

        cropMaskCtx.lineWidth = Number(cropBrush.value) || 20;
        cropMaskCtx.lineCap = 'round';
        cropMaskCtx.lineJoin = 'round';
        cropMaskCtx.strokeStyle = 'rgba(255,0,0,0.6)';

        cropMaskCtx.lineTo(x, y);
        cropMaskCtx.stroke();

        cropMaskCtx.beginPath();
        cropMaskCtx.moveTo(x, y);
      };

      cropMask.addEventListener('pointerdown', (e) => {
        cropDrawing = true;
        cropMask.setPointerCapture(e.pointerId);

        const [x, y] = cropXY(e);
        cropMaskCtx.beginPath();
        cropMaskCtx.moveTo(x, y);
        cropStroke(e);
      });

      cropMask.addEventListener('pointermove', (e) => {
        if (cropDrawing) cropStroke(e);
      });

      const endCropStroke = () => {
        cropDrawing = false;
        cropMaskCtx.beginPath();
      };
      cropMask.addEventListener('pointerup', endCropStroke);
      cropMask.addEventListener('pointercancel', endCropStroke);

      $(`${p}-crop-mask-clear`).onclick = () => {
        cropMaskCtx.clearRect(0, 0, cropMask.width, cropMask.height);
      };
    }

    // 칠한 부분이 있으면 흑백 PNG(dataURL), 없으면 null (= 마스크 미사용)
    const getCropMask = () => {
      if (!cropMask || !rect || !cropMask.width || !cropMask.height) return null;

      const { width, height } = cropMask;
      const src = cropMaskCtx.getImageData(0, 0, width, height).data;

      const output = document.createElement('canvas');
      output.width = width;
      output.height = height;
      const outCtx = output.getContext('2d');
      const outPixels = outCtx.createImageData(width, height);

      let any = false;
      for (let i = 0; i < src.length; i += 4) {
        const v = src[i + 3] > 30 ? 255 : 0;
        if (v) any = true;
        outPixels.data[i] = v;
        outPixels.data[i + 1] = v;
        outPixels.data[i + 2] = v;
        outPixels.data[i + 3] = 255;
      }

      if (!any) return null;

      outCtx.putImageData(outPixels, 0, 0);
      return output.toDataURL('image/png');
    };

    const reset = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      localMask = null;
      rect = null;
      rectStart = null;
      showCrop(null);
    };

    const getXY = (e) => {
      const r = canvas.getBoundingClientRect();
      return [
        (e.clientX - r.left) * canvas.width / r.width,
        (e.clientY - r.top) * canvas.height / r.height,
      ];
    };

    // ----- Load image -----
    $(`${p}-load_image`).onclick = async () => {
      try {
        const response = await fetch('api/load_image', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: $(`${p}-img_path`).value }),
        });

        const data = await response.json();
        if (!response.ok || !data.success) {
          throw new Error(data.error || 'Image load failed');
        }

        sourceImage = new Image();

        sourceImage.onload = () => {
          bg.width = canvas.width = sourceImage.width;
          bg.height = canvas.height = sourceImage.height;

          bg.getContext('2d').drawImage(sourceImage, 0, 0);
          reset();

          $(`${p}-mask-editor`).style.display = 'block';
          $(`${p}-orig-empty`).style.display = 'none';
          setStatus(isRect
            ? `이미지 로드 완료. 드래그로 영역을 선택하세요. (${UNIT}px 단위)`
            : '이미지 로드 완료.');
        };

        sourceImage.onerror = () => setStatus('이미지 로드 실패.');
        sourceImage.src = `data:image/png;base64,${data.image}`;
      } catch (e) {
        outEl.textContent = `failed: ${e.message}`;
      }
    };

    // ----- Brush (inpainting) -----
    const brushEl = $(`${p}-brush`);

    const draw = (e) => {
      const [x, y] = getXY(e);

      ctx.lineWidth = Number(brushEl && brushEl.value) || 20;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.strokeStyle = 'rgba(255,0,0,0.6)';

      ctx.lineTo(x, y);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(x, y);
    };

    // ----- Rect (detailer) -----
    // (a, b) = UNIT 개수. 긴 변이 ALLOW 가 되도록 확대했을 때
    // 짧은 변도 UNIT 의 정수배로 딱 떨어지는 조합만 허용.
    //   짧은변' = S*UNIT * ALLOW / (L*UNIT) = S*ALLOW/L  →  UNIT 배수여야 함
    const isExact = (a, b) => {
      const L = Math.max(a, b);
      const S = Math.min(a, b);
      return (S * ALLOW) % (L * UNIT) === 0;
    };

    // 드래그한 크기(px)에 가장 가까운, 영역 안에 들어가는 유효 조합
    const snapRect = (dw, dh, roomW, roomH) => {
      const maxA = Math.floor(roomW / UNIT);
      const maxB = Math.floor(roomH / UNIT);
      if (maxA < 1 || maxB < 1) return { w: 0, h: 0 };

      const ta = dw / UNIT;
      const tb = dh / UNIT;

      let best = null;
      let bestCost = Infinity;

      for (let a = 1; a <= maxA; a++) {
        for (let b = 1; b <= maxB; b++) {
          if (!isExact(a, b)) continue;
          const cost = (a - ta) ** 2 + (b - tb) ** 2;
          if (cost < bestCost) {
            bestCost = cost;
            best = [a, b];
          }
        }
      }

      return best ? { w: best[0] * UNIT, h: best[1] * UNIT } : { w: 0, h: 0 };
    };

    const drawRect = (a, b) => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const dx = b[0] - a[0];
      const dy = b[1] - a[1];

      // 시작점 고정, 드래그 방향으로만 늘어남
      const { w, h } = snapRect(
        Math.abs(dx),
        Math.abs(dy),
        dx >= 0 ? canvas.width - a[0] : a[0],
        dy >= 0 ? canvas.height - a[1] : a[1],
      );

      let x = Math.round(dx >= 0 ? a[0] : a[0] - w);
      let y = Math.round(dy >= 0 ? a[1] : a[1] - h);
      x = Math.max(0, Math.min(x, canvas.width - w));
      y = Math.max(0, Math.min(y, canvas.height - h));

      if (w > 0 && h > 0) {
        ctx.fillStyle = 'rgba(255,0,0,0.25)';
        ctx.fillRect(x, y, w, h);

        ctx.lineWidth = Math.max(2, canvas.width / 400);
        ctx.strokeStyle = 'rgba(255,0,0,0.95)';
        ctx.strokeRect(x, y, w, h);
      }

      return { x, y, w, h };
    };

    // ----- Pointer events -----
    canvas.addEventListener('pointerdown', (e) => {
      if (!sourceImage) return;

      drawing = true;
      canvas.setPointerCapture(e.pointerId);

      localMask = null;
      rect = null;

      if (isRect) {
        rectStart = getXY(e);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        setStatus('영역 선택 중...');
        return;
      }

      const [x, y] = getXY(e);
      ctx.beginPath();
      ctx.moveTo(x, y);
      draw(e);
      setStatus('마스크 수정 중...');
    });

    canvas.addEventListener('pointermove', (e) => {
      if (!drawing) return;

      if (isRect) {
        const r = drawRect(rectStart, getXY(e));
        setStatus(`영역 선택 중... ${r.w}×${r.h}`);
        showCrop(r);
      } else {
        draw(e);
      }
    });

    canvas.addEventListener('pointerup', (e) => {
      if (!drawing) return;
      drawing = false;

      if (isRect && rectStart) {
        const r = drawRect(rectStart, getXY(e));

        if (r.w >= UNIT && r.h >= UNIT) {
          rect = {
            x: Math.round(r.x),
            y: Math.round(r.y),
            w: Math.round(r.w),
            h: Math.round(r.h),
          };
          setStatus(`영역 선택 완료 (${rect.w}×${rect.h})`);
          showCrop(rect);
        } else {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          setStatus('영역이 너무 작아요. 다시 드래그하세요.');
          showCrop(null);
        }

        rectStart = null;
      }
    });

    canvas.addEventListener('pointercancel', () => {
      drawing = false;
      rectStart = null;

      if (isRect) {
        rect = null;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        setStatus('영역 선택이 취소되었습니다.');
        showCrop(null);
      }
    });

    // ----- Clear -----
    $(`${p}-mask-clear`).onclick = () => {
      reset();
      setStatus('초기화 완료.');
    };

    // ----- Finish mask (inpainting only) -----
    const finishBtn = $(`${p}-mask-finish`);
    if (finishBtn) {
      finishBtn.onclick = () => {
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
        setStatus('마스크 생성 완료.');
      };
    }

    setStatus(isRect
      ? `이미지를 로드한 뒤 드래그로 영역을 선택하세요. (${UNIT}px 단위)`
      : '이미지를 로드한 뒤 마스크를 그리고 Finish mask를 눌러주세요.');

    return {
      getMask: () => localMask,
      getRect: () => rect,
      getCropMask,
    };
  };

  // =========================================================
  // Progress / Console  (한 번에 한 작업만 실행)
  // =========================================================

  let busy = false;
  let runningMode = null;
  let pollTimer = null;

  const allRunBtns = () => Object.values(PREFIX).map((p) => $(`${p}-run`));

  const startPoll = (p) => {
    const bar = $(`${p}-bar`);
    const out = $(`${p}-out`);

    bar.style.display = 'block';
    bar.value = 0;
    bar.max = 1;

    pollTimer = setInterval(async () => {
      try {
        const r = await fetch('api/progress');
        const prog = await r.json();

        if (!pollTimer) return;

        if (prog.total > 0) {
          bar.max = prog.total;
          bar.value = prog.step;
          out.textContent = prog.step >= prog.total
            ? 'decoding / saving...'
            : `sampling ${prog.step}/${prog.total}`;
        } else {
          out.textContent = 'preparing...';
        }
      } catch (_) { /* ignore */ }
    }, 300);
  };

  const stopPoll = (p) => {
    if (pollTimer !== null) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
    $(`${p}-bar`).style.display = 'none';
  };

  // 로그는 실행 중인 탭의 콘솔에, 없으면 현재 탭의 콘솔에 붙임
  setInterval(async () => {
    try {
      const r = await fetch('api/logs');
      const data = await r.json();
      const lines = data.lines || [];
      if (!lines.length) return;

      const el = $(`${PREFIX[runningMode || activeMode]}-console`);

      const stick = el.scrollTop + el.clientHeight >= el.scrollHeight - 5;
      el.textContent += lines.join('\n') + '\n';
      if (stick) el.scrollTop = el.scrollHeight;
    } catch (_) { /* ignore */ }
  }, 500);

  // =========================================================
  // Generate (공통)
  // =========================================================

  // collect() → { params } 또는 { error }
  const initRun = (mode, collect, afterRun) => {
    const p = PREFIX[mode];
    const out = $(`${p}-out`);
    const result = $(`${p}-result`);

    $(`${p}-clear_out`).onclick = () => {
      out.textContent = '';
      $(`${p}-console`).textContent = '';
      result.innerHTML = '';
    };

    $(`${p}-run`).onclick = async () => {
      if (busy) return;

      const c = collect();
      if (c.error) {
        out.textContent = c.error;
        return;
      }

      const params = {
        mode,
        sampler: $(`${p}-sampler`).value,
        steps: $(`${p}-steps`).value,
        seed: $(`${p}-seed`).value,
        shift: $(`${p}-shift`).value,
        eta: $(`${p}-eta`).value,
        ...c.params,
      };

      busy = true;
      runningMode = mode;
      allRunBtns().forEach((b) => { b.disabled = true; });

      out.textContent = 'running...';
      result.innerHTML = '';
      startPoll(p);

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
        result.append(img);

        out.textContent = 'done';
      } catch (e) {
        out.textContent = `failed: ${e.message}`;
      } finally {
        stopPoll(p);
        busy = false;
        runningMode = null;
        allRunBtns().forEach((b) => { b.disabled = false; });
        if (afterRun) afterRun();
      }
    };
  };

  // =========================================================
  // Tab 1 — Text2img
  // =========================================================

  (() => {
    const p = PREFIX.txt2img;
    const tools = initTagTools(p);

    initRun('txt2img', () => ({
      params: {
        tags: $(`${p}-tags`).value,
        fixed: $(`${p}-fixed`).checked,
        nsfw: $(`${p}-nsfw`).value === 'true',
        cfg: $(`${p}-cfg`).value,
        H: $(`${p}-H`).value,
        W: $(`${p}-W`).value,
      },
    }), tools.loadHistory);
  })();

  // =========================================================
  // Tab 2 — Inpainting
  // =========================================================

  (() => {
    const p = PREFIX.inpainting;
    const tools = initTagTools(p);
    const editor = initEditor(p, 'mask');

    initRun('inpainting', () => {
      const imgPath = $(`${p}-img_path`).value;
      if (!imgPath.trim()) return { error: '이미지 경로를 입력해 주세요.' };

      const mask = editor.getMask();
      if (!mask) return { error: '마스크를 그린 뒤 Finish mask를 눌러주세요.' };

      return {
        params: {
          tags: $(`${p}-tags`).value,
          fixed: $(`${p}-fixed`).checked,
          nsfw: $(`${p}-nsfw`).value === 'true',
          cfg: $(`${p}-cfg`).value,
          img_path: imgPath,
          start_t: $(`${p}-start_t`).value,
          mask,
        },
      };
    }, tools.loadHistory);
  })();

  // =========================================================
  // Tab 3 — Detailer
  // =========================================================

  (() => {
    const p = PREFIX.detailer;
    const editor = initEditor(p, 'rect');
    initAutocomplete($(`${p}-tags`), $(`${p}-ac`));

    initRun('detailer', () => {
      const imgPath = $(`${p}-img_path`).value;
      if (!imgPath.trim()) return { error: '이미지 경로를 입력해 주세요.' };

      const rect = editor.getRect();
      if (!rect) return { error: '드래그로 사각형 영역을 선택해 주세요.' };

      return {
        params: {
          tags: $(`${p}-tags`).value,
          fixed: $(`${p}-fixed`).checked,
          nsfw: $(`${p}-nsfw`).value === 'true',
          cfg: $(`${p}-cfg`).value,
          img_path: imgPath,
          start_t: $(`${p}-start_t`).value,
          rect,
          mask: editor.getCropMask(),
        },
      };
    });
  })();

  // =========================================================
  // 초기 탭 복원: URL 해시 → 마지막 탭 → txt2img
  // =========================================================

  (() => {
    let initial = location.hash.slice(1);
    if (!PREFIX[initial]) {
      try { initial = localStorage.getItem('mmdit.tab'); } catch (_) { initial = null; }
    }
    setTab(PREFIX[initial] ? initial : 'txt2img');
  })();
})();
