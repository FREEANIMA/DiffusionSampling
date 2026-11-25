/* sampling.js — txt2img 전용 */
import { initCommon, collectCommon, toSamplingRequest, postJSON } from '../common/common.js';

import { API_BASE } from '../common/common.js';
const API_URL = `${API_BASE}/api/sampling`;

export async function init() {
    await initCommon({ size: true, onGenerate });
}

// 요청만 보냄 (응답 처리는 아직 안 함). prompt 하나당 요청 하나, 순서대로
// body = SamplingRequest { prompt, sigma_curve, cfg_curve, eta_curve, seed, steps, width, height }
async function onGenerate() {
    const btn = document.getElementById('t2i-generate');
    btn.disabled = true;                // 보내는 동안 중복 클릭 방지
    try {
        const params = collectCommon();
        for (const prompt of params.prompts) {
            const body = toSamplingRequest(params, prompt);
            console.log('send', body);
            await postJSON(API_URL, body);
            // TODO: 응답 이미지가 오면 setResult(i, url) 호출
        }
    } catch (err) {
        console.error(err);
        alert(err.message);
    } finally {
        btn.disabled = false;
    }
}