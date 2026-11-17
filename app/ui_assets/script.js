(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);

  // null-safe helpers
  const setDisplay = (id, v) => {
    const el = $(id);
    if (el) el.style.display = v;
  };
  const setCols = (id, v) => {
    const el = $(id);
    if (el) el.style.gridTemplateColumns = v;
  };

  const modeEl = $('mode');

  // =========================================================
  // Autocomplete
  // =========================================================

  const TAGS = window.TAG_LIST || [];
  const key = (s) =>
    s.trim().toLowerCase().replace(/[\s_]+/g, ' ');

  const KEYED = TAGS.map((t) => [t, key(t)]);

  const acEl = $('ac');
  const tagsEl = $('tags');

  let acItems = [];
  let acIdx = -1;

  const lev = (a, b) => {
    const m = a.length;
    const n = b.length;

    if (Math.abs(m - n) > 2) return 99;

    let prev = Array.from(
      { length: n + 1 },
      (_, j) => j
    );

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

  const currentToken = () => {
    const v = tagsEl.value;
    const pos = tagsEl.selectionStart ?? v.length;
    const start = v.lastIndexOf(',', pos - 1) + 1;

    return {
      start,
      end: pos,
      text: v.slice(start, pos),
    };
  };

  const escapeHtml = (s) =>
    String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');

  const search = (raw) => {
    const q = key(raw);
    if (!q) return [];

    const pre = [];
    const sub = [];

    for (const [t, k] of KEYED) {
      if (k.startsWith(q)) {
        pre.push(t);
      } else if (k.includes(q)) {
        sub.push(t);
      }
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

  const closeAc = () => {
    acEl.style.display = 'none';
    acIdx = -1;
  };

  const renderAc = () => {
    if (!acItems.length) {
      closeAc();
      return;
    }

    acEl.innerHTML = acItems.map((t, i) =>
      `<div class="ac-item${i === acIdx ? ' on' : ''}" data-i="${i}">
        ${escapeHtml(t)}
      </div>`
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

    if (e.key === 'ArrowDown') {
      acIdx = (acIdx + 1) % acItems.length;
      renderAc();
      e.preventDefault();
    } else if (e.key === 'ArrowUp') {
      acIdx = (acIdx - 1 + acItems.length) % acItems.length;
      renderAc();
      e.preventDefault();
    } else if (
      (e.key === 'Enter' || e.key === 'Tab') &&
      acIdx >= 0
    ) {
      pick(acItems[acIdx]);
      e.preventDefault();
    } else if (e.key === 'Escape') {
      closeAc();
    }
  });

  acEl.addEventListener('mousedown', (e) => {
    const i = e.target.dataset.i;
    if (i !== undefined) {
      pick(acItems[+i]);
      e.preventDefault();
    }
  });

  tagsEl.addEventListener('blur', () => {
    setTimeout(closeAc, 100);
  });

  // =========================================================
  // Tag History
  // =========================================================

  const histEl = $('tag_history');
  let histItems = [];

  const renderHistory = () => {
    if (!histItems.length) {
      histEl.innerHTML =
        '<span class="history-empty">기록 없음</span>';
      return;
    }

    histEl.innerHTML = histItems.map((it, i) =>
      `<button type="button"
        class="history-item"
        data-i="${i}"
        title="${escapeHtml(it.input)}">
        <span class="h-text">
          ${escapeHtml(it.input || '(빈 입력)')}
        </span>
        <span class="h-count">
          +${it.tags.length}
        </span>
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
    } catch (_) {
      /* ignore */
    }
  };

  histEl.addEventListener('click', (e) => {
    const btn = e.target.closest('.history-item');
    if (!btn) return;

    const it = histItems[+btn.dataset.i];
    if (!it) return;

    const base = it.input
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean);

    tagsEl.value = [...new Set([...base, ...it.tags])].join(', ');
    $('fixed').checked = true;
  });

  $('history_refresh').onclick = loadHistory;

  loadHistory();

  // =========================================================
  // Tag Tree (type별 태그 브라우저)
  // 서버 응답 형식: { type: [[tag, count], ...] }
  // =========================================================

  const treeEl = $('tag_tree');
  const treeSearchEl = $('tree_search');
  const TREE_LIMIT = 200;

  let treeData = {};   // { type: [[tag, count], ...] }

  const addTag = (tag) => {
    const cur = tagsEl.value
      .split(',').map((s) => s.trim()).filter(Boolean);
    if (!cur.includes(tag)) cur.push(tag);
    tagsEl.value = cur.join(', ') + ', ';
    tagsEl.focus();
  };

  const fillGroup = (box, tags, q) => {
    const list = q
      ? tags.filter(([t]) => key(t).includes(q))
      : tags;
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

      // 검색 중이면 매칭 있는 type만, 자동으로 펼침
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

  const loadTree = async () => {
    try {
      const r = await fetch('api/tag_tree');
      if (!r.ok) throw new Error('Tree request failed');
      treeData = (await r.json()).tree || {};
      renderTree();
    } catch (_) {
      treeEl.innerHTML = '<span class="history-empty">불러오기 실패</span>';
    }
  };

  let treeTimer = null;
  treeSearchEl.addEventListener('input', () => {
    clearTimeout(treeTimer);
    treeTimer = setTimeout(renderTree, 150);   // 디바운스
  });

  treeEl.addEventListener('click', (e) => {
    const btn = e.target.closest('.tree-tag');
    if (btn) addTag(btn.dataset.tag);
  });

  loadTree();

  // =========================================================
  // Image / Mask / Detailer Rect
  // =========================================================

  let localMask = null;
  let rect = null;
  let rectStart = null;

  let sourceImage = null;
  let drawing = false;

  const isDetailer = () => modeEl.value === 'detailer';

  const canvas = $('mask-canvas');
  const ctx = canvas.getContext('2d');

  const getXY = (e) => {
    const r = canvas.getBoundingClientRect();
    return [
      (e.clientX - r.left) * canvas.width / r.width,
      (e.clientY - r.top) * canvas.height / r.height,
    ];
  };

  // ---------------------------------------------------------
  // Load Image
  // ---------------------------------------------------------

  $('load_image').onclick = async () => {
    try {
      const response = await fetch('api/load_image', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          path: $('img_path').value,
        }),
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.error || 'Image load failed');
      }

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
          ? '이미지 로드 완료. 드래그로 영역을 선택하세요. (64px 단위)'
          : '이미지 로드 완료.';

        localMask = null;
        rect = null;
        rectStart = null;
      };

      sourceImage.onerror = () => {
        $('mask-status').textContent = '이미지 로드 실패.';
      };

      sourceImage.src = `data:image/png;base64,${data.image}`;
    } catch (e) {
      $('out').textContent = `failed: ${e.message}`;
    }
  };

  // ---------------------------------------------------------
  // Inpainting Brush
  // ---------------------------------------------------------

  const draw = (e) => {
    const [x, y] = getXY(e);

    ctx.lineWidth = Number($('brush').value) || 20;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = 'rgba(255,0,0,0.6)';

    ctx.lineTo(x, y);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(x, y);
  };

  // ---------------------------------------------------------
  // Detailer Rectangle (64px 배수로 스냅)
  // ---------------------------------------------------------

  const UNIT = 64; // patch_size(2) * compression_ratio(32)

  // len을 UNIT 배수로 반올림하되, 이미지 안에 들어가는 최대치(room)로 제한
  const snapLen = (len, room) => {
    const maxLen = Math.floor(room / UNIT) * UNIT;
    if (maxLen < UNIT) return 0;
    return Math.min(maxLen, Math.max(UNIT, Math.round(len / UNIT) * UNIT));
  };

  const drawRect = (a, b) => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const dx = b[0] - a[0];
    const dy = b[1] - a[1];

    // 시작점 고정, 드래그 방향으로만 늘어남
    const w = snapLen(Math.abs(dx), dx >= 0 ? canvas.width - a[0] : a[0]);
    const h = snapLen(Math.abs(dy), dy >= 0 ? canvas.height - a[1] : a[1]);

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

  // ---------------------------------------------------------
  // Pointer Events
  // ---------------------------------------------------------

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

    if (isDetailer()) {
      const r = drawRect(rectStart, getXY(e));
      $('mask-status').textContent = `영역 선택 중... ${r.w}×${r.h}`;
    } else {
      draw(e);
    }
  });

  const endDraw = (e) => {
    if (!drawing) return;
    drawing = false;

    if (isDetailer() && rectStart) {
      const r = drawRect(rectStart, getXY(e));

      if (r.w >= UNIT && r.h >= UNIT) {
        rect = {
          x: Math.round(r.x),
          y: Math.round(r.y),
          w: Math.round(r.w),
          h: Math.round(r.h),
        };
        $('mask-status').textContent =
          `영역 선택 완료 (${rect.w}×${rect.h})`;
      } else {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        $('mask-status').textContent =
          '영역이 너무 작아요. 다시 드래그하세요.';
      }

      rectStart = null;
    }
  };

  canvas.addEventListener('pointerup', endDraw);

  canvas.addEventListener('pointercancel', () => {
    drawing = false;
    rectStart = null;

    if (isDetailer()) {
      rect = null;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      $('mask-status').textContent = '영역 선택이 취소되었습니다.';
    }
  });

  // ---------------------------------------------------------
  // Clear Mask / Rectangle
  // ---------------------------------------------------------

  $('mask-clear').onclick = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    localMask = null;
    rect = null;
    rectStart = null;

    $('mask-status').textContent = '초기화 완료.';
  };

  // ---------------------------------------------------------
  // Finish Inpainting Mask
  // ---------------------------------------------------------

  $('mask-finish').onclick = () => {
    if (isDetailer()) return;

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

  // =========================================================
  // UI Mode
  // =========================================================

  const START_T_DEFAULT = {
    inpainting: '0.9',
    detailer: '0.3',
  };

  const applyMode = () => {
    const m = modeEl.value;

    const isT2I = m === 'txt2img';
    const isDet = m === 'detailer';

    // tree-card: 보일 땐 CSS의 display:flex를 그대로 쓰도록 빈 문자열
    setDisplay('tree-card', isDet ? 'none' : '');
    // 그리드 열을 CSS 클래스로 전환 (인라인 스타일은 미디어쿼리를 무시함)
    const grid = document.querySelector('.app-grid');
    if (grid) grid.classList.toggle('no-tree', isDet);

    setDisplay('txt2img-box', isT2I ? 'block' : 'none');
    setDisplay('inpaint-box', isT2I ? 'none' : 'block');

    const title = $('inpaint-title');
    if (title) title.textContent = isDet ? 'Detailer' : 'Inpainting';

    setDisplay('detailer-box', isDet ? 'block' : 'none');
    setDisplay('brush-field', isDet ? 'none' : 'block');
    setDisplay('mask-finish', isDet ? 'none' : 'inline-block');

    // Detailer에서는 사용하지 않는 항목 숨김
    setDisplay('tags-field', isDet ? 'none' : 'block');
    setDisplay('history-field', isDet ? 'none' : 'block');
    setDisplay('nsfw-field', isDet ? 'none' : 'block');
    setDisplay('cfg-field', isDet ? 'none' : 'block');

    // 그리드 열 조정
    setCols('sampling-grid-2', isDet ? '1fr' : '');
    setCols('sampling-grid-3', isDet ? '1fr 1fr' : '');

    const st = $('start_t');
    if (st && START_T_DEFAULT[m]) st.value = START_T_DEFAULT[m];
  };

  modeEl.onchange = () => {
    applyMode();

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    localMask = null;
    rect = null;
    rectStart = null;

    $('mask-status').textContent = isDetailer()
      ? '드래그로 영역을 선택하세요. (64px 단위)'
      : '마스크를 그린 뒤 Finish mask를 눌러주세요.';
  };

  try {
    applyMode();
  } catch (e) {
    console.error('applyMode failed:', e);
  }

  // =========================================================
  // Clear Output
  // =========================================================

  $('clear_out').onclick = () => {
    $('out').textContent = '';
    $('console').textContent = '';

    const result = $('result');
    if (result) result.innerHTML = '';
  };

  // =========================================================
  // Progress
  // =========================================================

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

          $('out').textContent =
            p.step >= p.total
              ? 'decoding / saving...'
              : `sampling ${p.step}/${p.total}`;
        } else {
          $('out').textContent = 'preparing...';
        }
      } catch (_) {
        /* ignore */
      }
    }, 300);
  };

  const stopPoll = () => {
    if (pollTimer !== null) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
    $('bar').style.display = 'none';
  };

  // =========================================================
  // Console
  // =========================================================

  const consoleEl = $('console');

  setInterval(async () => {
    try {
      const r = await fetch('api/logs');
      const data = await r.json();
      const lines = data.lines || [];

      if (!lines.length) return;

      const stick =
        consoleEl.scrollTop + consoleEl.clientHeight >=
        consoleEl.scrollHeight - 5;

      consoleEl.textContent += lines.join('\n') + '\n';

      if (stick) {
        consoleEl.scrollTop = consoleEl.scrollHeight;
      }
    } catch (_) {
      /* ignore */
    }
  }, 500);

  // =========================================================
  // Generate
  // =========================================================

  $('run').onclick = async () => {
    const mode = modeEl.value;

    const params = {
      mode,
      sampler: $('sampler').value,
      steps: $('steps').value,
      seed: $('seed').value,
    };

    // txt2img
    if (mode === 'txt2img') {
      Object.assign(params, {
        tags: $('tags').value,
        fixed: $('fixed').checked,
        nsfw: $('nsfw').value === 'true',
        cfg: $('cfg').value,
        H: $('H').value,
        W: $('W').value,
      });
    }

    // inpainting
    if (mode === 'inpainting') {
      Object.assign(params, {
        tags: $('tags').value,
        fixed: $('fixed').checked,
        nsfw: $('nsfw').value === 'true',
        cfg: $('cfg').value,
        img_path: $('img_path').value,
        start_t: $('start_t').value,
        mask: localMask,
      });

      if (!localMask) {
        $('out').textContent =
          '마스크를 그린 뒤 Finish Mask를 눌러주세요.';
        return;
      }
    }

    // detailer
    if (mode === 'detailer') {
      Object.assign(params, {
        img_path: $('img_path').value,
        start_t: $('start_t').value,
        rect: rect,
      });

      if (!rect) {
        $('out').textContent =
          '드래그로 사각형 영역을 선택해 주세요.';
        return;
      }
    }

    // validate image path
    if (mode === 'inpainting' || mode === 'detailer') {
      if (!params.img_path.trim()) {
        $('out').textContent = '이미지 경로를 입력해 주세요.';
        return;
      }
    }

    const runBtn = $('run');
    if (runBtn.disabled) return;

    runBtn.disabled = true;
    $('out').textContent = 'running...';

    startPoll();

    const result = $('result');
    if (result) result.innerHTML = '';

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

      if (result) result.append(img);

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