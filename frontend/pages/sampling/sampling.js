/* sampling.js — txt2img 전용 */
import { initCommon, collectCommon, toSamplingRequest, postJSON, API_BASE } from '../common/common.js';

const API_URL = `${API_BASE}/api/sampling`;

export async function init() {
    await initCommon({ size: true, onGenerate });
}

// 요청 한 번에 prompts 배열 전체를 보냄 (1개여도 배열)
// body = SamplingRequest { prompts, sigma_curve, cfg_curve, eta_curve, seed, steps, width, height }
async function onGenerate() {
    const btn = document.getElementById('t2i-generate');
    btn.disabled = true;                // 보내는 동안 중복 클릭 방지
    try {
        const params = collectCommon();
        const body = toSamplingRequest(params);     // prompts: ["prompt1", "prompt2", ...]
        console.log('send', body);
        await postJSON(API_URL, body);
        // TODO: 응답으로 이미지 URL 배열이 오면 순서대로 setResult(i, url) 호출
    } catch (err) {
        console.error(err);
        alert(err.message);
    } finally {
        btn.disabled = false;
    }
}