import torch

def sigma_schedular(step, device, shift=3.0):
    # 1. 0~1 사이를 균등하게 뽑음 (학습의 Uniform 분포와 일치)
    # 학습 코드의 bin_idx + rand()와 동일한 효과를 내도록 1.0 -> 0.0으로 설정
    t = torch.linspace(1.0, 0.0, step + 1, device=device)

    # 2. 학습 스케줄러와 동일한 Shift 공식 적용
    s = shift
    sigmas = s * t / (1 + (s - 1) * t)
    
    # 3. 학습 때 사용한 범위(0.001 ~ 0.999)로 클램핑
    sigmas = sigmas.clamp(0.001, 0.999)
    
    # 4. [중요] 마지막 시그마는 반드시 0.001(또는 0)이어야 함
    # clamp로 인해 마지막 값이 0.001로 고정되었는지 확인
    sigmas[-1] = 0.001 
    
    return sigmas