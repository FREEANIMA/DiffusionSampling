import torch

def sigma_schedular(step, device, shift=1.0):
    # MMDiT(SD3/FLUX) 계열은 시그마가 0~1 사이로 정규화되어 있음
    sigma_max = 1.0
    sigma_min = 0.001 # 0으로 두면 수학적 오류(0 나누기)가 발생할 수 있어 아주 작은 값 사용
    
    # Karras 공식의 rho (곡률). 보통 7.0을 쓰지만, 
    # 인자로 받은 shift를 rho로 활용해 튜닝할 수 있게 함.
    # (shift=1.0이면 기본 Karras, shift를 높이면 낮은 시그마에 더 집중)
    rho = 7.0 if shift == 1.0 else shift 
    
    # 1. Karras 공식을 이용해 시그마 생성
    step_indices = torch.arange(step, dtype=torch.float32, device=device)
    
    # t_i = (sigma_max^(1/rho) + i/(N-1) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho
    t = (sigma_max ** (1 / rho) + 
         step_indices / (step - 1) * 
         (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    
    # 2. 마지막 스텝은 완전히 노이즈를 제거하기 위해 0 추가
    sigmas = torch.cat([t, torch.zeros_like(t[:1])])
    
    return sigmas