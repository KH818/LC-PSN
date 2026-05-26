import numpy as np
from scipy import linalg, signal

# 안테나 배열 파라미터 (노트북 설정값 반영)
M = 8   # 안테나 소자 개수

def ULA_action_vector(theta, m=M):
    """
    Uniform Linear Array(ULA)의 Steering Vector(조향 벡터)를 계산합니다.
    theta: 입사각 (radians)
    m: 안테나 소자 개수
    """
    array = np.linspace(0, m, m, endpoint=False) 
    # 신호의 입사각에 따른 위상차 계산: exp(-j * pi * n * sin(theta))
    return np.exp(- 1j * np.pi * array * np.sin(theta))

def construct_signal(thetas, snr, snapshots, m=M, var_signal_power=1, var_noise=1):
    """
    비상관(Non-coherent) 신호를 생성합니다.
    """
    #y(t) = A(theta)x(t) + n(t)
    d = len(thetas)
    # 신호 생성 (Complex Gaussian)
    sig = np.sqrt(var_signal_power) * np.sqrt(10 ** (snr / 10)) * \
          (np.random.randn(d, snapshots) + 1j * np.random.randn(d, snapshots))
    
    # Steering Matrix A 구성
    A = np.array([ULA_action_vector(t, m) for t in thetas])
    
    # 가우시안 노이즈 생성
    noise = np.sqrt(var_noise) * (np.random.randn(m, snapshots) + 1j * np.random.randn(m, snapshots))
    
    # 수신 신호 X = AS + N
    return np.dot(A.T, sig) + noise, sig

def construct_coherent_signal(thetas, snr, snapshots, m=M, var_signal_power=1, var_noise=1):
    """
    상관(Coherent) 신호를 생성합니다. (모든 소스가 동일한 신호원을 공유)
    """
    d = len(thetas)
    # 단일 신호원을 생성하여 반복 사용
    sig = np.sqrt(var_signal_power) * (10 ** (snr / 10)) * \
          (np.random.randn(1, snapshots) + 1j * np.random.randn(1, snapshots))
    sig = np.repeat(sig, d, axis=0)

    A = np.array([ULA_action_vector(t, m) for t in thetas])
    noise = np.sqrt(var_noise) * (np.random.randn(m, snapshots) + 1j * np.random.randn(m, snapshots))

    return np.dot(A.T, sig) + noise, sig

def classic_music(incident, continuum, sources=None, m=M):
    """
    전통적인 MUSIC 알고리즘입니다.
    incident: 수신된 신호 행렬
    continuum: 스캔할 각도 범위 (예: -pi/2 ~ pi/2)
    sources: 신호원의 개수 (None일 경우 고윳값 클러스터링으로 추정)
    """
    covariance = np.cov(incident) #공분산 행렬
    eigenvalues, eigenvectors = linalg.eig(covariance) #EVD
    
    # 고윳값 크기순 정렬
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    if sources is None:
        # 고윳값 분석을 통한 신호원 개수 추정 (노트북의 cluster 로직 참고)
        # 여기서는 단순화를 위해 소스 개수가 주어지지 않으면 최소 고윳값을 기준으로 분리
        threshold = 1.25
        n = np.sum(np.abs(eigenvalues) < np.abs(eigenvalues[-1]) + threshold)
        d = m - n
    else:
        d = sources

    # Noise Subspace (En) 추출
    En = eigenvectors[:, d:]

    num_samples = continuum.shape[1]
    spectrum = np.zeros(num_samples)
    
    # 각도별 스펙트럼 계산
    for i in range(num_samples):
        a = ULA_action_vector(continuum[0, i], m)
        # P_music = 1 / (a^H * En * En^H * a)
        spectrum[i] = 1. / np.abs(a.conj().T @ En @ En.conj().T @ a)

    # 피크 검출을 통해 DOA 추정
    peaks, _ = signal.find_peaks(spectrum)
    est_doas = peaks[np.argsort(spectrum[peaks])[-d:]]

    return est_doas, spectrum, d

def beamformer(incident, continuum, m=M):
    """
    기본적인 Beamformer (Bartlett) 알고리즘입니다.
    """
    covariance = np.cov(incident)
    num_samples = continuum.shape[1]
    spectrum = np.zeros(num_samples)

    for i in range(num_samples):
        a = ULA_action_vector(continuum[0, i], m)
        spectrum[i] = np.abs(a.conj().T @ covariance @ a) / linalg.norm(a)**2

    return spectrum

def quantize_complex_1bit(incident, normalize=False):
    """
    Complex 1-bit quantization (naive):
      z = sign(Re{x}) + j*sign(Im{x})
    incident: shape (m, snapshots), complex
    normalize: if True, divide by sqrt(2) (optional scaling)
    """
    z = np.sign(np.real(incident)) + 1j * np.sign(np.imag(incident))
    if normalize:
        z = z / np.sqrt(2)
    return z


def onebit_cov_arcsine(z):
    """
    Arcsine/Van-Vleck (Bussgang) covariance correction for 1-bit quantized complex signals.

    Input:
      z: 1-bit complex signal, shape (m, snapshots)
         expected entries roughly in {±1 ± j} (optionally scaled)

    Output:
      R_x_hat: corrected (unquantized-like, normalized) covariance estimate, shape (m, m)

    Notes:
    - For 1-bit quantization of a zero-mean jointly Gaussian vector,
      the correlation of the quantized signal relates to the correlation of the original
      via an arcsin law. A common practical "inverse" uses:
         Re{R_x} ≈ sin( (π/2) * Re{R_z_corr} )
         Im{R_x} ≈ sin( (π/2) * Im{R_z_corr} )
      where R_z_corr is the correlation coefficient matrix of z (unit diagonal).
    - This produces a covariance-like matrix up to a scale; for MUSIC, scale is not critical.
    """
    # sample covariance of quantized signal
    Rz = np.cov(z)

    # convert to correlation coefficient matrix (unit diagonal)
    d = np.sqrt(np.clip(np.real(np.diag(Rz)), 1e-12, None))
    Dinv = np.diag(1.0 / d)
    Cz = Dinv @ Rz @ Dinv  # normalized so diag ~ 1

    # apply "inverse arcsine" (sine transform)
    Rx_re = np.sin((np.pi / 2.0) * np.real(Cz))
    Rx_im = np.sin((np.pi / 2.0) * np.imag(Cz))
    Rx = Rx_re + 1j * Rx_im

    # enforce Hermitian (numerical)
    Rx = 0.5 * (Rx + Rx.conj().T)
    return Rx


def onebit_music_naive(incident, continuum, sources=None, m=M, threshold=1.25):
    """
    1-bit MUSIC (naive):
    - quantize incident to 1-bit complex
    - run the same EVD + MUSIC as classic_music, with unknown-K via threshold clustering
    """
    z = quantize_complex_1bit(incident, normalize=False)

    covariance = np.cov(z)
    eigenvalues, eigenvectors = linalg.eig(covariance)

    # sort by magnitude descending
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    if sources is None:
        n = np.sum(np.abs(eigenvalues) < np.abs(eigenvalues[-1]) + threshold)
        d = m - n
    else:
        d = sources

    En = eigenvectors[:, d:]

    num_samples = continuum.shape[1]
    spectrum = np.zeros(num_samples)
    for i in range(num_samples):
        a = ULA_action_vector(continuum[0, i], m)
        spectrum[i] = 1. / np.abs(a.conj().T @ En @ En.conj().T @ a)

    peaks, _ = signal.find_peaks(spectrum)
    est_doas = peaks[np.argsort(spectrum[peaks])[-d:]]
    return est_doas, spectrum, d


def onebit_music_arcsine(incident, continuum, sources=None, m=M, threshold=1.25):
    """
    1-bit MUSIC (arcsine/Bussgang corrected):
    - quantize incident to 1-bit complex
    - estimate corrected covariance via arcsine/Van-Vleck inverse (sine transform)
    - EVD + MUSIC as usual
    """
    z = quantize_complex_1bit(incident, normalize=False)
    covariance = onebit_cov_arcsine(z)

    eigenvalues, eigenvectors = linalg.eig(covariance)

    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    if sources is None:
        n = np.sum(np.abs(eigenvalues) < np.abs(eigenvalues[-1]) + threshold)
        d = m - n
    else:
        d = sources

    En = eigenvectors[:, d:]

    num_samples = continuum.shape[1]
    spectrum = np.zeros(num_samples)
    for i in range(num_samples):
        a = ULA_action_vector(continuum[0, i], m)
        spectrum[i] = 1. / np.abs(a.conj().T @ En @ En.conj().T @ a)

    peaks, _ = signal.find_peaks(spectrum)
    est_doas = peaks[np.argsort(spectrum[peaks])[-d:]]
    return est_doas, spectrum, d