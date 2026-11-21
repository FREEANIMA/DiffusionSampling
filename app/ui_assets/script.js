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

  // JSON이 아닌 응답(프록시의 HTML 에러 페이지 등)이 와도 원인을 알 수 있게 함
  const fetchJson = async (url, opts) => {
    const r = await fetch(url, opts);
    const text = await r.text();
    try {
      return { ok: r.ok, status: r.status, data: JSON.parse(text) };
    } catch (_) {
      throw new Error(`HTTP ${r.status} (JSON 아님: ${text.slice(0, 60).replace(/\s+/g, ' ')})`);
    }
  };

  const sleep = (ms) => new Promise((res) => setTimeout(res, ms));

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
  // 2-레이어 페인터
  //   재생성(빨강) / 보존(파랑) 레이어를 따로 보관한다.
  //   한쪽을 칠하면 반대쪽 레이어에서는 지워져서 서로 겹치지 않는다.
  //   (나중에 칠한 쪽이 이김)
  // =========================================================

  const REGEN_COLOR = '#ff3b3b';
  const KEEP_COLOR = '#3b6bff';

  const createPainter = ({ view, brushEl, modeBox }) => {
    const vctx = view.getContext('2d');
    const regen = document.createElement('canvas');
    const keep = document.createElement('canvas');
    const rctx = regen.getContext('2d', { willReadFrequently: true });
    const kctx = keep.getContext('2d', { willReadFrequently: true });

    let mode = 'regen';
    let last = null;

    const render = () => {
      vctx.clearRect(0, 0, view.width, view.height);
      vctx.globalAlpha = 0.55;
      vctx.drawImage(regen, 0, 0);
      vctx.drawImage(keep, 0, 0);
      vctx.globalAlpha = 1;
    };

    // 크기가 바뀌면 칠한 내용은 초기화됨
    const resize = (w, h) => {
      view.width = regen.width = keep.width = w;
      view.height = regen.height = keep.height = h;
      render();
    };

    const clear = () => {
      rctx.clearRect(0, 0, regen.width, regen.height);
      kctx.clearRect(0, 0, keep.width, keep.height);
      render();
    };

    const xy = (e) => {
      const b = view.getBoundingClientRect();
      return [
        (e.clientX - b.left) * view.width / b.width,
        (e.clientY - b.top) * view.height / b.height,
      ];
    };

    const paint = (ctx, op, color, a, b, w) => {
      ctx.save();
      ctx.globalCompositeOperation = op;
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = w;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      ctx.beginPath();
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(b[0], b[1], w / 2, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    };

    const stroke = (a, b) => {
      const w = Number(brushEl && brushEl.value) || 20;

      if (mode === 'regen') {
        paint(rctx, 'source-over', REGEN_COLOR, a, b, w);
        paint(kctx, 'destination-out', '#000', a, b, w);
      } else if (mode === 'keep') {
        paint(kctx, 'source-over', KEEP_COLOR, a, b, w);
        paint(rctx, 'destination-out', '#000', a, b, w);
      } else {
        paint(rctx, 'destination-out', '#000', a, b, w);
        paint(kctx, 'destination-out', '#000', a, b, w);
      }
      render();
    };

    view.addEventListener('pointerdown', (e) => {
      view.setPointerCapture(e.pointerId);
      last = xy(e);
      stroke(last, last);
    });

    view.addEventListener('pointermove', (e) => {
      if (!last) return;
      const cur = xy(e);
      stroke(last, cur);
      last = cur;
    });

    const end = () => { last = null; };
    view.addEventListener('pointerup', end);
    view.addEventListener('pointercancel', end);

    modeBox.addEventListener('click', (e) => {
      const btn = e.target.closest('.mode-btn');
      if (!btn) return;
      mode = btn.dataset.mode;
      modeBox.querySelectorAll('.mode-btn').forEach((b) => b.classList.toggle('active', b === btn));
    });

    // 칠한 곳이 있으면 흑백 PNG(dataURL), 없으면 null
    const exportLayer = (layer) => {
      const { width, height } = layer;
      if (!width || !height) return null;

      const src = layer.getContext('2d').getImageData(0, 0, width, height).data;

      const out = document.createElement('canvas');
      out.width = width;
      out.height = height;
      const octx = out.getContext('2d');
      const px = octx.createImageData(width, height);

      let any = false;
      for (let i = 0; i < src.length; i += 4) {
        const v = src[i + 3] > 127 ? 255 : 0;
        if (v) any = true;
        px.data[i] = v;
        px.data[i + 1] = v;
        px.data[i + 2] = v;
        px.data[i + 3] = 255;
      }

      if (!any) return null;

      octx.putImageData(px, 0, 0);
      return out.toDataURL('image/png');
    };

    return {
      resize,
      clear,
      exportMasks: () => ({ regen: exportLayer(regen), keep: exportLayer(keep) }),
    };
  };

  // =========================================================
  // Image editor (inpainting 마스크 / detailer 사각형 + crop 마스크) — 탭별 인스턴스
  // kind: 'mask' | 'rect'
  // =========================================================

  // ui.py 가 window.DETAILER_CFG 로 내려줌 (없으면 기본값)
  const CFG = window.DETAILER_CFG || {};
  const UNIT = CFG.unit || 32;          // patch_size * compression_ratio
  const ALLOW = CFG.allowable || 1024 ;  // allowable_size

  const initEditor = (p, kind) => {
    const isRect = kind === 'rect';

    const canvas = $(`${p}-mask-canvas`);
    const bg = $(`${p}-bg-canvas`);
    const ctx = canvas.getContext('2d');
    const statusEl = $(`${p}-mask-status`);
    const outEl = $(`${p}-out`);

    let sourceImage = null;
    let drawing = false;
    let rect = null;
    let rectStart = null;

    const setStatus = (t) => { statusEl.textContent = t; };

    // 칠하는 캔버스: inpainting = 원본 위, detailer = crop 위
    const painter = createPainter({
      view: isRect ? $(`${p}-crop-mask-canvas`) : canvas,
      brushEl: $(`${p}-brush`),
      modeBox: $(`${p}-paint-mode`),
    });

    // ----- Crop 확대 미리보기 (detailer) -----
    const cropCanvas = isRect ? $(`${p}-crop-canvas`) : null;
    const cropStage = isRect ? $(`${p}-crop-stage`) : null;
    const cropEmpty = isRect ? $(`${p}-crop-empty`) : null;
    const cropInfo = isRect ? $(`${p}-crop-info`) : null;

    const showCrop = (r) => {
      if (!cropCanvas) return;

      if (!r || r.w <= 0 || r.h <= 0 || !sourceImage) {
        cropStage.style.display = 'none';
        cropEmpty.style.display = '';
        cropInfo.textContent = '';
        return;
      }

      cropCanvas.width = r.w;
      cropCanvas.height = r.h;
      painter.resize(r.w, r.h);   // 크기가 바뀌므로 칠한 내용은 초기화됨
      cropCanvas.getContext('2d').drawImage(sourceImage, r.x, r.y, r.w, r.h, 0, 0, r.w, r.h);

      cropStage.style.display = 'block';
      cropEmpty.style.display = 'none';
      const k = ALLOW / Math.max(r.w, r.h);
      cropInfo.textContent = `${r.w}×${r.h} → ${Math.round(r.w * k)}×${Math.round(r.h * k)}`;
    };

    const reset = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      rect = null;
      rectStart = null;
      showCrop(null);
      painter.clear();
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
          bg.width = sourceImage.width;
          bg.height = sourceImage.height;

          if (isRect) {
            canvas.width = sourceImage.width;
            canvas.height = sourceImage.height;
          } else {
            painter.resize(sourceImage.width, sourceImage.height);
          }

          bg.getContext('2d').drawImage(sourceImage, 0, 0);
          reset();

          $(`${p}-mask-editor`).style.display = 'block';
          $(`${p}-orig-empty`).style.display = 'none';
          setStatus(isRect
            ? `이미지 로드 완료. 드래그로 영역을 선택하세요. (${UNIT}px 단위)`
            : '이미지 로드 완료. 재생성(빨강) / 보존(파랑)으로 칠하세요. 안 칠하면 전체 재생성.');
        };

        sourceImage.onerror = () => setStatus('이미지 로드 실패.');
        sourceImage.src = `data:image/png;base64,${data.image}`;
      } catch (e) {
        outEl.textContent = `failed: ${e.message}`;
      }
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
        ctx.fillStyle = 'rgba(255,255,255,0.18)';
        ctx.fillRect(x, y, w, h);

        ctx.lineWidth = Math.max(2, canvas.width / 400);
        ctx.strokeStyle = 'rgba(255,255,255,0.95)';
        ctx.strokeRect(x, y, w, h);
      }

      return { x, y, w, h };
    };

    if (isRect) {
      canvas.addEventListener('pointerdown', (e) => {
        if (!sourceImage) return;

        drawing = true;
        canvas.setPointerCapture(e.pointerId);

        rect = null;
        rectStart = getXY(e);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        setStatus('영역 선택 중...');
      });

      canvas.addEventListener('pointermove', (e) => {
        if (!drawing) return;

        const r = drawRect(rectStart, getXY(e));
        setStatus(`영역 선택 중... ${r.w}×${r.h}`);
        showCrop(r);
      });

      canvas.addEventListener('pointerup', (e) => {
        if (!drawing) return;
        drawing = false;

        if (rectStart) {
          const r = drawRect(rectStart, getXY(e));

          if (r.w >= UNIT && r.h >= UNIT) {
            rect = {
              x: Math.round(r.x),
              y: Math.round(r.y),
              w: Math.round(r.w),
              h: Math.round(r.h),
            };
            setStatus(`영역 선택 완료 (${rect.w}×${rect.h}) — 가운데 crop에서 칠하세요`);
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
        rect = null;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        setStatus('영역 선택이 취소되었습니다.');
        showCrop(null);
      });

      $(`${p}-crop-mask-clear`).onclick = () => painter.clear();
    }

    // ----- Clear -----
    $(`${p}-mask-clear`).onclick = () => {
      if (isRect) reset();
      else painter.clear();
      setStatus('초기화 완료.');
    };

    setStatus(isRect
      ? `이미지를 로드한 뒤 드래그로 영역을 선택하세요. (${UNIT}px 단위)`
      : '이미지를 로드한 뒤 재생성(빨강) / 보존(파랑)으로 칠하세요. 안 칠하면 전체 재생성.');

    return {
      getMasks: () => painter.exportMasks(),   // { regen, keep }  (칠한 게 없으면 null)
      getRect: () => rect,
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
  // Result viewer (번호 버튼으로 이미지 전환)
  // =========================================================

  const clearResult = (p) => {
    $(`${p}-result`).innerHTML = '';
    $(`${p}-pager`).innerHTML = '';
  };

  const showImages = (p, srcs, seeds) => {
    clearResult(p);
    if (!srcs.length) return;

    const result = $(`${p}-result`);
    const pager = $(`${p}-pager`);

    const img = document.createElement('img');
    img.onerror = () => { $(`${p}-out`).textContent = '이미지 로드 실패 (api/image 응답 확인)'; };
    result.append(img);

    const select = (i) => {
      img.src = srcs[i];
      img.title = seeds && seeds[i] !== undefined ? `seed ${seeds[i]}` : `#${i + 1}`;
      [...pager.children].forEach((b, j) => b.classList.toggle('active', j === i));
    };

    if (srcs.length > 1) {
      srcs.forEach((_, i) => {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'pager-btn';
        b.textContent = String(i + 1);
        b.onclick = () => select(i);
        pager.append(b);
      });
    }

    select(0);
  };

  // =========================================================
  // Generate (공통)
  // 요청은 job_id 만 받고 바로 끝나고, 결과는 api/result/<job_id> 를 폴링해서 받음
  // → 오래 걸려도 프록시 타임아웃에 걸리지 않는다.
  // =========================================================

  // 일시적인 네트워크/프록시 오류인지 (재시도 대상)
  const isTransient = (e) =>
    e instanceof TypeError || String(e.message).startsWith('HTTP');

  const MAX_POLL_FAILS = 30;

  // collect() → { params } 또는 { error }
  const initRun = (mode, collect, afterRun) => {
    const p = PREFIX[mode];
    const out = $(`${p}-out`);

    $(`${p}-clear_out`).onclick = () => {
      out.textContent = '';
      $(`${p}-console`).textContent = '';
      clearResult(p);
    };

    $(`${p}-run`).onclick = async () => {
      if (busy) return;

      const c = collect();
      if (c.error) {
        out.textContent = c.error;
        return;
      }

      // batch 입력칸이 없는 탭이 있어도 안전하게 1장으로
      const batchEl = $(`${p}-batch`);
      const batch = batchEl
        ? Math.max(1, Math.min(parseInt(batchEl.value, 10) || 1, 8))
        : 1;

      const params = {
        mode,
        sampler: $(`${p}-sampler`).value,
        steps: $(`${p}-steps`).value,
        seed: $(`${p}-seed`).value,
        shift: $(`${p}-shift`).value,
        eta: $(`${p}-eta`).value,
        eta_mode: $(`${p}-eta_mode`).value,
        frame_save: $(`${p}-frame_save`).checked,
        batch,
        ...c.params,
      };

      busy = true;
      runningMode = mode;
      allRunBtns().forEach((b) => { b.disabled = true; });

      out.textContent = 'running...';
      clearResult(p);
      startPoll(p);

      try {
        const start = await fetchJson('api/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(params),
        });
        if (!start.ok || !start.data.success) {
          throw new Error(start.data.error || 'Generation failed');
        }

        const jobId = start.data.job_id;
        let data = null;
        let fails = 0;

        // 결과가 나올 때까지 폴링 (일시적인 네트워크/프록시 오류는 무시하고 재시도)
        while (!data) {
          await sleep(1000);

          try {
            const res = await fetchJson(`api/result/${jobId}`);
            fails = 0;

            if (res.data.status === 'running') continue;
            if (!res.data.success) throw new Error(res.data.error || 'Generation failed');

            data = res.data;
          } catch (e) {
            if (isTransient(e) && ++fails < MAX_POLL_FAILS) continue;
            throw e;
          }
        }

        // 서버가 urls 를 주면 그대로, 구버전(base64)도 호환
        const srcs = data.urls
          || (data.images || (data.image ? [data.image] : []))
               .map((b) => `data:image/png;base64,${b}`);
        showImages(p, srcs, data.seeds);

        out.textContent = srcs.length > 1 ? `done (${srcs.length} images)` : 'done';
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
  //   regen_mask(빨강) / keep_mask(파랑): 칠한 게 없으면 null
  //   둘 다 null 이면 전체 재생성
  // =========================================================

  (() => {
    const p = PREFIX.inpainting;
    const tools = initTagTools(p);
    const editor = initEditor(p, 'mask');

    initRun('inpainting', () => {
      const imgPath = $(`${p}-img_path`).value;
      if (!imgPath.trim()) return { error: '이미지 경로를 입력해 주세요.' };

      const m = editor.getMasks();

      return {
        params: {
          tags: $(`${p}-tags`).value,
          fixed: $(`${p}-fixed`).checked,
          nsfw: $(`${p}-nsfw`).value === 'true',
          cfg: $(`${p}-cfg`).value,
          img_path: imgPath,
          start_t: $(`${p}-start_t`).value,
          regen_mask: m.regen,
          keep_mask: m.keep,
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

      const m = editor.getMasks();

      return {
        params: {
          tags: $(`${p}-tags`).value,
          fixed: $(`${p}-fixed`).checked,
          nsfw: $(`${p}-nsfw`).value === 'true',
          cfg: $(`${p}-cfg`).value,
          img_path: imgPath,
          start_t: $(`${p}-start_t`).value,
          rect,
          regen_mask: m.regen,
          keep_mask: m.keep,
        },
      };
    });
  })();

  // =========================================================
  // Size controls (Text2img): 슬라이더 + 숫자 입력
  //   - 두 변 모두 1024 이하 → 자유
  //   - 한 변이라도 1024 초과 → 큰 변을 2048로 보정
  // =========================================================

  (() => {
    const p = PREFIX.txt2img;
    const SIZE_MIN = 64;
    const SIZE_MAX = 2048;
    const FREE_MAX = 1024;
    const STEP = 32;

    const els = {
      H: { num: $(`${p}-H`), range: $(`${p}-H-range`) },
      W: { num: $(`${p}-W`), range: $(`${p}-W-range`) },
    };

    const snap = (v) => {
      v = Number(v);
      if (!Number.isFinite(v)) v = FREE_MAX;
      v = Math.round(v / STEP) * STEP;
      return Math.max(SIZE_MIN, Math.min(SIZE_MAX, v));
    };

    const write = (k, v) => {
      els[k].num.value = v;
      els[k].range.value = v;
    };

    // edited: 방금 사용자가 건드린 변 ('H' | 'W')
    const apply = (edited) => {
      const other = edited === 'H' ? 'W' : 'H';

      let a = snap(els[edited].num.value);
      let b = snap(els[other].num.value);

      // 한 변이라도 1024 초과인데 2048인 변이 없으면 → 큰 변을 2048로
      if (Math.max(a, b) > FREE_MAX && Math.max(a, b) < SIZE_MAX) {
        if (a >= b) a = SIZE_MAX;
        else b = SIZE_MAX;
      }

      write(edited, a);
      write(other, b);
    };

    ['H', 'W'].forEach((k) => {
      // 슬라이더: 움직이는 즉시 보정
      els[k].range.addEventListener('input', () => {
        els[k].num.value = els[k].range.value;
        apply(k);
      });

      // 숫자 입력: 타이핑 중에는 건드리지 않고, 확정(Enter/blur/스피너)될 때 보정
      els[k].num.addEventListener('change', () => apply(k));
      els[k].num.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') apply(k);
      });
    });

    // 초기값 정리
    apply('H');
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