/* sampling.js — inpainting 전용 */
import { initCommon, collectCommon, setResult } from '../common/common.js';

export async function init() {
    await initCommon({ size: true, onGenerate });
}

async function onGenerate() {
    const params = collectCommon();     // prompts, steps, cfg, seed, width, height, sigmas, cfgs, etas, styles
    console.log('generate', params);
    // TODO: 백엔드 응답으로 setResult(i, url) 호출
}