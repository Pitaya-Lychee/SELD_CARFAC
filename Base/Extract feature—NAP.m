clear all
clc
close all
clear variables
%------------------------���������ź�----------------------------------------
%---------------------------------------------------------------------------
wav='test_0_desc_30_300.wav';
[test_signal,fs]=audioread(wav);

% ֱ�ӿ��Ƕ������������
n_ears = 1;
%----------------------------------------------------------------------------
%---------------------------��ʼ��CAR��IHC��AGC����---------------------------
CF_CAR_params = struct( ...
    'velocity_scale', 0.1, ...  % P254�еķ����Ժ����ٶ�scale
    'v_offset', 0.04, ...  % P254�е�offset
    'min_zeta', 0.10, ... % ��С�������Ӧ�
    'max_zeta', 0.35, ... % ����������Ӧ�
    'first_pole_theta', 0.85*pi, ... % ��һ������ǶȵĦ�
    'zero_ratio', sqrt(2), ... % �㼫�����
    'high_f_damping_compression', 0.5, ... % 0 to 1 ��Ƶ����ѹ��
    'ERB_per_step', 0.5, ... 
    'min_pole_Hz', 30, ...
    'ERB_break_freq', 165.3, ... 
    'ERB_Q', 1000/(24.7*4.37));  

just_hwr=0;
one_cap=1;
CF_IHC_params = struct( ...
    'just_hwr', just_hwr, ...       
    'one_cap', one_cap, ...   
    'tau_lpf', 0.000080, ...  % P266����˫��ƽ���˲���
    'tau_out', 0.0005, ...    % P266�е�һ����ͨ�˲�����ʱ�䳣����ͨ����ȥ�����ź��Թ��ɸ�ͨ�˲���,���������ڻ���Ĥ������ƽ������ʱ����20hz���µ�Ƶ��
    'tau_in', 0.010, ...      % P266����IHC�Զ�������ƵĻ�·�˲�����ʱ�䳣��
    'ac_corner_Hz', 20);  % �����ڻ���Ĥ������ƽ������ʱ����20hz���µ�Ƶ��

CF_AGC_params = struct( ...
    'n_stages', 4, ...  % P269����4�׵�ƽ���˲���
    'time_constants', 0.002 * 4.^(0:3), ... % ʱ�䳣��0.002,0.008,0.032,0.128
    'AGC_stage_gain', 2, ...  % ��һ��������������Ȩ��2
    'decimation', [8, 2, 2, 2], ...  % AGC����ʱ��㣬���˺�Ϊ[8,16,32,64]
    'AGC1_scales', 1.0 * sqrt(2).^(0:3), ...   %  �ϵ����϶�ƽ������
    'AGC2_scales', 1.65 * sqrt(2).^(0:3), ... %  �϶����ϵ�ƽ������
    'AGC_mix_coeff', 0.5);  %�������ϵ��
%----------------------------------------------------------------------------
%-------------------------��ͨ�����Լ���Ӧ������Ƶ��---------------------------
pole_Hz = CF_CAR_params.first_pole_theta * fs / (2*pi);  % ͨ����һ����ĽǶȦ���ü����Ƶ��
n_ch = 0;
while pole_Hz > CF_CAR_params.min_pole_Hz
  n_ch = n_ch + 1;
  pole_Hz = pole_Hz - CF_CAR_params.ERB_per_step * ...
    (CF_CAR_params.ERB_break_freq + pole_Hz) / CF_CAR_params.ERB_Q;
end
% ERB=24.7*(1+4.37*pole_Hz/1000),ERB_per_step=0.5,ERB_break_freq=165.3,ERB_Q=1000/(24.7*4.37)
% ���ͨ����n_ch=71

% ��ÿ��ͨ��������Ƶ��pole_freqs�Ӵ�С�����:(71,1)
pole_freqs = zeros(n_ch, 1);
pole_Hz = CF_CAR_params.first_pole_theta * fs / (2*pi);
for ch = 1:n_ch
  pole_freqs(ch) = pole_Hz;
  pole_Hz = pole_Hz - CF_CAR_params.ERB_per_step * ...
    (CF_CAR_params.ERB_break_freq + pole_Hz) / CF_CAR_params.ERB_Q;
end
max_channels_per_octave = log(2) / log(pole_freqs(1)/pole_freqs(2));% �����Ƶ���˶�:12.2709
%----------------------------------------------------------------------------
%-----------------------------CAR_coeffs-------------------------------------
CAR_coeffs = struct( ...
  'n_ch', n_ch, ...   % 71
  'velocity_scale', CF_CAR_params.velocity_scale, ...  % P254 ��NLF�е�scale=0.1
  'v_offset', CF_CAR_params.v_offset ...   % P254 ��NLF�е�offset=0.04
  );
CAR_coeffs.r1_coeffs = zeros(n_ch, 1); % ����뾶�������������й�
CAR_coeffs.a0_coeffs = zeros(n_ch, 1);%  a0Ϊ����ǵ�����
CAR_coeffs.c0_coeffs = zeros(n_ch, 1); % c0Ϊ����ǵ�����
CAR_coeffs.h_coeffs = zeros(n_ch, 1);  % h���ڿ�������뼫���Ƶ�ʱ���
CAR_coeffs.g0_coeffs = zeros(n_ch, 1); % g���ڵ���������
f = CF_CAR_params.zero_ratio^2 - 1; 
% zero_ratio��ʾΪ�㼫�����sqrt(2),f��Ŀ����Ϊ�����hֵ����h=c0*f
% P245,�����Ƶ�ʽ��ȼ���Ƶ�ʸ߳���Լ�����Ƶ��ʱ��f=1��h=c0
theta = pole_freqs .* (2 * pi / fs);
x = theta/pi; 
% ���ÿ������Ƶ������Ӧ�ļ���Ƕ�theta=�ȣ�������ת��Ϊ����x=w
c0 = sin(theta); % c0Ϊ����ǵ�����
a0 = cos(theta); % a0Ϊ����ǵ�����
ff = CF_CAR_params.high_f_damping_compression; % ������Ϊ0 to 1���˴�����0.5����Ƶ����ѹ����
zr_coeffs = pi * (x - ff * x.^3); 

max_zeta = CF_CAR_params.max_zeta;   
% 'max_zeta', 0.35, ����������Ӧ�
CAR_coeffs.r1_coeffs = (1 - zr_coeffs .* max_zeta);  
% r1����������������µİ뾶
min_zeta = CF_CAR_params.min_zeta;
% 'min_zeta', 0.10����С�������Ӧ�
min_zetas = min_zeta + 0.25* ...
  (((CF_CAR_params.ERB_break_freq + pole_Hz) / CF_CAR_params.ERB_Q)./ pole_freqs - min_zeta);
% ���ݵ�Ч����������ͨ������������С����
CAR_coeffs.zr_coeffs = zr_coeffs .* ...
  (max_zeta - min_zetas);
% �����Ը�����drz
CAR_coeffs.a0_coeffs = a0;
CAR_coeffs.c0_coeffs = c0;
h = c0 .* f;  
CAR_coeffs.h_coeffs = h;
relative_undamping = ones(n_ch, 1);  % �˴���Ϊδ����AGC��Ԫ������b=0����r=r1+drz*(1-b)�е�(1-b)��Ϊ1
% CAR_coeffs.g0_coeffs = CARFAC_Stage_g(CAR_coeffs, relative_undamping);
r1 = CAR_coeffs.r1_coeffs;  % r1����������������µİ뾶
a0 = CAR_coeffs.a0_coeffs;  
c0 = CAR_coeffs.c0_coeffs;
h  = CAR_coeffs.h_coeffs;  % h���ڿ�������뼫���Ƶ�ʱ���
zr = CAR_coeffs.zr_coeffs;  %drz ��Ը�����
r  = r1 + zr .* relative_undamping;  % r=r1+drz*(1-b)
g  = (1 - 2*r.*a0 + r.^2) ./ (1 - 2*r.*a0 + h.*r.*c0 + r.^2); % ����P246��g�Ĺ�ʽ�ó�

% AGC_coeffs = CARFAC_DesignAGC(AGC_params, fs, n_ch)
%----------------------------------------------------------------------------
%-----------------------------AGC_coeffs-------------------------------------
n_AGC_stages = CF_AGC_params.n_stages; % ƽ���˲�������4
% 'AGC1_scales', 1.0 * sqrt(2).^(0:3), ...   % AGC1ͨ�����ϵ����϶�ƽ��
% 'AGC2_scales', 1.65 * sqrt(2).^(0:3), ...  % AGC2ͨ�����϶����ϵ�ƽ��
AGC1_scales = CF_AGC_params.AGC1_scales;
AGC2_scales = CF_AGC_params.AGC2_scales;
decim = 1;
total_DC_gain = 0;
AGC_coeffs = struct([]);
for stage = 1:n_AGC_stages
  AGC_coeffs(stage).n_ch = n_ch;
  AGC_coeffs(stage).n_AGC_stages = n_AGC_stages;
  AGC_coeffs(stage).AGC_stage_gain = CF_AGC_params.AGC_stage_gain;
  % 'time_constants', 0.002 * 4.^(0:3), ... % ʱ�䳣��0.002,0.008,0.032,0.128
  % 'AGC_stage_gain', 2, ...  % ��һ��������������Ȩ��2
  % 'decimation', [8, 2, 2, 2], ...  % ���ݵ��˸���ʱ���Ϊ[8,16,32,64]
  % 'AGC1_scales', 1.0 * sqrt(2).^(0:3)  AGC1ͨ�����ϵ����϶�ƽ��
  % 'AGC2_scales', 1.65 * sqrt(2).^(0:3) AGC2ͨ�����϶����ϵ�ƽ��
  AGC_coeffs(stage).decimation = CF_AGC_params.decimation(stage);
  tau = CF_AGC_params.time_constants(stage);  % ʱ�䳣��(s)
  decim = decim * CF_AGC_params.decimation(stage);  % ����ʱ��㣬����ݽ����۳� [8,16,32,64]
  % AGC_epsilon��ʾÿһ��������Ҫ���������룬��⣺���ݸ���ʱ��㡢ʱ�䳣��������������Ҫ��ǰ��೤ʱ�����������ƽ��
  AGC_coeffs(stage).AGC_epsilon = 1 - exp(-decim / (tau * fs)); 
  % һ��ʱ�䳣����ƽ������Ч����:
  %����ÿ��AGC����ʱ���������AGC��ͨ�˲���״̬���У�������ʱ��ƽ���ĳ�ʱ�������ж����Ч������
  ntimes = tau * (fs / decim);  % decim[8,16,32,64]

  % ȷ��������Ӧ��Ŀ����ɢ(����)���ӳ�(��ֵ)��ΪҪ���n�εķֲ���
  delay = (AGC2_scales(stage) - AGC1_scales(stage)) / ntimes; 
  spread_sq = (AGC1_scales(stage)^2 + AGC2_scales(stage)^2) / ntimes; 
  % ��ü���λ�ã��Ը��õ�ƥ��ÿ��������[[���ηֲ�]]��Ԥ����չ���ӳ�
  u = 1 + 1 / spread_sq;  
  p = u - sqrt(u^2 - 1);   
  dp = delay * (1 - 2*p +p^2)/2;
  polez1 = p - dp;
  polez2 = p + dp;
  AGC_coeffs(stage).AGC_polez1 = polez1;
  AGC_coeffs(stage).AGC_polez2 = polez2;

  n_taps = 0;
  done = 0;
  n_iterations = 1;
  if spread_sq == 0
    n_iterations = 0;
    n_taps = 3;
    done = 1;
  end
  while ~done
    switch n_taps
      case 0
        % 3��FIR:
        n_taps = 3;
      case 3
        % 5��FIR
        n_taps = 5;
      case 5
        n_iterations = n_iterations + 1;
        if n_iterations > 4
          n_iteration = -1;  % Signal to use IIR instead.
        end
      otherwise
        error('Bad n_taps in CARFAC_DesignAGC');
    end
    % [AGC_spatial_FIR, done] = Design_FIR_coeffs(n_taps, spread_sq, delay, n_iterations);
    % �������ͨ������Ȩ�أ�����ֵΪAGC_spatial_FIR=[CL,1-CL-CR,CR]
    % ͨ��n�ε�����Сƽ���ֲ��ľ�ֵ�ͷ���:
    mean_delay = delay / n_iterations;
    delay_variance = spread_sq / n_iterations;
    switch n_taps
      case 3
        a = (delay_variance + mean_delay*mean_delay - mean_delay) / 2;
        b = (delay_variance + mean_delay*mean_delay + mean_delay) / 2;
        AGC_spatial_FIR = [a, 1 - a - b, b];
        done = AGC_spatial_FIR(2) >= 0.25;
      case 5
        a = ((delay_variance + mean_delay*mean_delay)*2/5 - mean_delay*2/3) / 2;
        b = ((delay_variance + mean_delay*mean_delay)*2/5 + mean_delay*2/3) / 2;
        AGC_spatial_FIR = [a/2, 1 - a - b, b/2];
        done = AGC_spatial_FIR(2) >= 0.15;
      otherwise
        error('Bad n_taps in AGC_spatial_FIR');
    end   
  end
  AGC_coeffs(stage).AGC_spatial_iterations = n_iterations; % �ռ��������
  AGC_coeffs(stage).AGC_spatial_FIR = AGC_spatial_FIR;% �������Ȩ��
  AGC_coeffs(stage).AGC_spatial_n_taps = n_taps; % n_taps��FIR�˲���

  % �ۻ����н׶ε�ֱ�����棬ռ�׶����棺
  total_DC_gain = total_DC_gain + CF_AGC_params.AGC_stage_gain^(stage-1);

  if stage == 1
    AGC_coeffs(stage).AGC_mix_coeffs = 0;
  else
    AGC_coeffs(stage).AGC_mix_coeffs = CF_AGC_params.AGC_mix_coeff / ...
      (tau * (fs / decim));  % tau * (fs / decim)=ntimes һ��ʱ�䳣����ƽ������Ч����
  end
end

% ����1��detect_scale����ΪAGC�˲���ֱ������ĵ�����
AGC_coeffs(1).detect_scale = 1 / total_DC_gain;

% IHC_coeffs = CARFAC_DesignIHC(IHC_params, fs, n_ch)
%----------------------------------------------------------------------------
%---------------------IHC_coeffs--------just_hwr=0;one_cap=1;----------------
x_in1=10;
a = 0.175;
set1 = x_in1 > -a;
zz1 = x_in1(set1)+a;
conductance1 = zeros(size(x_in1));
conductance1(set1) = zz1.^3 ./ (zz1.^3 + zz1.^2 + 0.1);  
ro = 1 / conductance1;  
% CARFAC_Detect�����ǽ�P266ͼ18-7�е�g����� 
% ro = 1/CARFAC_Detect(10);
c = CF_IHC_params.tau_out / ro; % tau_outΪ0.5ms�ķŵ磨�����ʱ�䳣�� 
ri = CF_IHC_params.tau_in / c;  % tau_inΪ10ms�洢��磨���룩ʱ�䳣����Ϊ�Զ����������LPFʱ�䳣��
saturation_output = 1 / (2*ro + ri); % �����µ����

% �ٿ���û���ź�����ʱ����IHCģ�͵�ƽ��:
x_in2=0;
set2 = x_in2 > -a;
zz2 = x_in2(set2)+a;
conductance2 = zeros(size(x_in2));
conductance2(set2) = zz2.^3 ./ (zz2.^3 + zz2.^2 + 0.1);  
r0 = 1 / conductance2; 
% r0 = 1/CARFAC_Detect(0);
current = 1 / (ri + r0);  % ֱ��q
cap_voltage = 1 - current * ri;  % ���޵�ѹv
IHC_coeffs = struct( ...
  'n_ch', n_ch, ...
  'just_hwr', 0, ...
  'lpf_coeff', 1 - exp(-1/(CF_IHC_params.tau_lpf * fs)), ...
  'out_rate', ro / (CF_IHC_params.tau_out * fs), ...
  'in_rate', 1 / (CF_IHC_params.tau_in * fs), ...
  'one_cap', CF_IHC_params.one_cap, ...
  'output_gain', 1/ (saturation_output - current), ... % ��һ��
  'rest_output', current / (saturation_output - current), ...
  'rest_cap', cap_voltage);
% ����/��֤�ĵ�ͨ��״̬:
IHC_state = struct( ...
  'cap_voltage', IHC_coeffs.rest_cap, ...
  'lpf1_state', 0, ...
  'lpf2_state', 0, ...
  'ihc_accum', 0);
IHC_coeffs.ac_coeff = 2 * pi * CF_IHC_params.ac_corner_Hz / fs;

%-----------------------------------------------------------------------------
%----------------����CF�ṹ�彫�ϱ����õ��Ľṹ��Ͳ��������ȥ------------------
for ear = 1:n_ears
  ears(ear).CAR_coeffs = CAR_coeffs;
  ears(ear).AGC_coeffs = AGC_coeffs;
  ears(ear).IHC_coeffs = IHC_coeffs;
end

CF = struct( ...
  'fs', fs, ...
  'max_channels_per_octave', max_channels_per_octave, ...
  'CAR_params', CF_CAR_params, ...
  'AGC_params', CF_AGC_params, ...
  'IHC_params', CF_IHC_params, ...
  'n_ch', n_ch, ...
  'pole_freqs', pole_freqs, ...
  'ears', ears, ...
  'n_ears', n_ears );

%----------------------------------------------------------------------------------
%---------------��ʼ�����������е���ʱ״̬����  CF = CARFAC_Init(CF);----------------
for ear = 1:n_ears
    n_ch = CF.ears(ear).CAR_coeffs.n_ch;
    CF.ears(ear).CAR_state = struct(...
      'z1_memory', zeros(n_ch, 1), ...
      'z2_memory', zeros(n_ch, 1), ...
      'zA_memory', zeros(n_ch, 1), ...
      'zB_memory', CF.ears(ear).CAR_coeffs.zr_coeffs, ...% dzr:CAR_coeffs.zr_coeffs = zr_coeffs .*(max_zeta - min_zetas);
      'dzB_memory', zeros(n_ch, 1), ...
      'zY_memory', zeros(n_ch, 1), ...
      'g_memory', CF.ears(ear).CAR_coeffs.g0_coeffs, ...% p251ͼ17-1�е�����g
      'dg_memory', zeros(n_ch, 1) ...
      );
    % CAR_Init_State������z1_memory��z2_memory��zA_memory��zB_memory����0����dzB_memory
    % zY_memory��g_memory����0����dg_memory

    n_ch = CF.ears(ear).IHC_coeffs.n_ch;
    CF.ears(ear).IHC_state = struct(...
      'ihc_accum', zeros(n_ch, 1), ...
      'cap_voltage', CF.ears(ear).IHC_coeffs.rest_cap * ones(n_ch, 1), ...
      'lpf1_state', CF.ears(ear).IHC_coeffs.rest_output * ones(n_ch, 1), ...
      'lpf2_state', CF.ears(ear).IHC_coeffs.rest_output * ones(n_ch, 1), ...
      'ac_coupler', zeros(n_ch, 1) ...
      );

     n_ch = CF.ears(ear).AGC_coeffs(1).n_ch; % ȡͨ����
     n_AGC_stages = CF.ears(ear).AGC_coeffs.n_AGC_stages;% ȡƽ���˲����Ľ���
     CF.ears(ear).AGC_state = struct([]);
    for stage = 1:n_AGC_stages
         CF.ears(ear).AGC_state(stage).AGC_memory = zeros(n_ch, 1);
         CF.ears(ear).AGC_state(stage).input_accum = zeros(n_ch, 1);
         CF.ears(ear).AGC_state(stage).decim_phase = 0;  % integer decimator phase
    end
  % CF.ears(ear).IHC_coeffs
  % IHC_Init_State������ihc_accum��cap_voltage����0����lpf1_state����0����lpf2_state����0����ac_coupler
  % AGC_Init_State������ÿ��ƽ���˲�����AGC_memory��input_accum��decim_phase=0
end
%----------------------------------------------------------------------------------
%-------------------------------------��֡��---------------------------------------
% agc_plot_fig_num = 10;
%[CF_struct, nap_decim, nap, BM, ohc, agc] = CARFAC_Run(CF_struct, test_signal,agc_plot_fig_num);
[n_samp,n_ears] = size(test_signal);
n_ch = CF.n_ch;
BM = zeros(n_samp,n_ch,n_ears);
ohc = zeros(n_samp,n_ch,n_ears);
agc = zeros(n_samp,n_ch,n_ears);
nap = zeros(n_samp,n_ch,n_ears);
seglen = 256; % ��֡��Ϊ160   882��20msһ֡(fs=44100) 882*
seg_move = 80;
step = seglen - seg_move;
n_segs = floor((n_samp-seg_move)/(seglen - seg_move))
decim_naps = zeros(n_segs,CF.n_ch,CF.n_ears);
if n_ears~=CF.n_ears
    error('bad number of input_waves channels passed to CARFAC_Run')
end
%----------------------------------------------------------------------------------
%--------------------------------------��֡����------------------------------------
for seg_num = 1:n_segs
    if seg_num == n_segs
    % The last segement may be short of seglen, but do it anyway:
        k_range = (step*(seg_num - 1) + seglen +1):n_samp;
    else
        k_range = seg_num * step+(1:seglen);
    end
    size1 = size(k_range);
    size1 = size1(2);
    win = hamming(size1);
    % [naps, CF, BM, seg_ohc, seg_agc] = CARFAC_Run_Segment(CF, input_waves, open_loop)
    %  [seg_naps, CF, seg_BM, seg_ohc, seg_agc] = CARFAC_Run_Segment(CF, test_signal(k_range, :), open_loop);
    % test_signal(k_range, :)Ϊÿһ֡������
    open_loop = 0;
    % do_BM = 1;  ��ΪҪ��ʾBMͼ������do_BM��Ϊ1
    input_waves = test_signal(k_range,:).*win;
    [n_samp1,n_ears] = size(input_waves);  % ��ʱn_sampΪÿһ֡�ĳ���
    if n_ears ~= CF.n_ears
        error('bad number of input_waves channels passed to CARFAC_Run')
    end
    seg_naps = zeros(n_samp1, n_ch, n_ears); 
    seg_BM = zeros(n_samp1, n_ch, n_ears);  % (�źų��ȣ�71��n_ears)
    seg_ohc = zeros(n_samp1, n_ch, n_ears);
    seg_agc = zeros(n_samp1, n_ch, n_ears);
    
    for k = 1:n_samp1
  % at each time step, possibly handle multiple channels  ��ÿ��ʱ�䲽���У����ܴ�����ͨ��
      for ear = 1:n_ears
   %--------------------------------------------------------------------------------------
   %------------����CAR��״̬��car_out�Ƿ���P251��ͼ17-1��DOHC�����Y----------------------
        %  [car_out, CF.ears(ear).CAR_state] = CARFAC_CAR_Step( ...
        %   input_waves(k, ear), CF.ears(ear).CAR_coeffs, CF.ears(ear).CAR_state);
        g = CF.ears(ear).CAR_state.g_memory+CF.ears(ear).CAR_state.dg_memory;
        zB = CF.ears(ear).CAR_state.zB_memory + CF.ears(ear).CAR_state.dzB_memory;
        zA = CF.ears(ear).CAR_state.zA_memory;
        v = CF.ears(ear).CAR_state.z2_memory - zA;
        nlf = 1 ./ (1 + (v * CAR_coeffs.velocity_scale + CAR_coeffs.v_offset) .^ 2 ); % p254
        r = CAR_coeffs.r1_coeffs + zB .* nlf; % p255�󼫵�뾶���˴���r=r1+drz*NLF(��δ��AGC�������1-b��)
        zA = CF.ears(ear).CAR_state.z2_memory;
        z1 = r .* (CAR_coeffs.a0_coeffs .* ...  % z1 = z1 + inputs; ͼ17-1��ͨ·
        CF.ears(ear).CAR_state.z1_memory - CAR_coeffs.c0_coeffs .* CF.ears(ear).CAR_state.z2_memory);
        z2 = r .* (CAR_coeffs.c0_coeffs .* ...  % ͼ17-1��ͨ·
        CF.ears(ear).CAR_state.z1_memory + CAR_coeffs.a0_coeffs .* CF.ears(ear).CAR_state.z2_memory);
        zY = CAR_coeffs.h_coeffs .* z2;    % �������h������ͨ·
        in_out = input_waves(k,ear);
        for ch = 1:length(zY)
          z1(ch) = z1(ch) + in_out;
          in_out = g(ch) * (in_out + zY(ch));  % P251ͼ17-1�����Y=g*(x+h)
          zY(ch) = in_out;
        end
        CF.ears(ear).CAR_state.z1_memory = z1;
        CF.ears(ear).CAR_state.z2_memory = z2;
        CF.ears(ear).CAR_state.zA_memory = zA;  % ��ʾ��ͨ·���������Ҫ��h�����
        CF.ears(ear).CAR_state.zB_memory = zB;  % drz
        CF.ears(ear).CAR_state.zY_memory = zY;
        CF.ears(ear).CAR_state.g_memory = g;
        car_out = zY;
        
    %--------------------------------------------------------------------------------------
    %------����IHC��״̬����DOHC�����Y��Ϊ���룬ihc_out�Ƿ���P266��ͼP18-7��DIHC�����NAP----
        %   [ihc_out, CF.ears(ear).IHC_state] = CARFAC_IHC_Step( ...
        %   car_out, CF.ears(ear).IHC_coeffs, CF.ears(ear).IHC_state);
        ac_diff = car_out - CF.ears(ear).IHC_state.ac_coupler; 
        CF.ears(ear).IHC_state.ac_coupler = CF.ears(ear).IHC_state.ac_coupler + CF.ears(ear).IHC_coeffs.ac_coeff * ac_diff;
        % conductance = CARFAC_Detect(ac_diff); P266�벨������NLFģ�飬conductanceΪg
        set = ac_diff > -a;
        zz = ac_diff(set)+a;
        conductance = zeros(size(ac_diff));
        conductance(set) = zz.^3 ./ (zz.^3 + zz.^2 + 0.1);
        ihc_out = conductance .* CF.ears(ear).IHC_state.cap_voltage;  % p266ͼ18-7 y=g*v
        CF.ears(ear).IHC_state.cap_voltage = CF.ears(ear).IHC_state.cap_voltage - ihc_out .* CF.ears(ear).IHC_coeffs.out_rate + ...
        (1 - CF.ears(ear).IHC_state.cap_voltage) .* CF.ears(ear).IHC_coeffs.in_rate;
        ihc_out = ihc_out * CF.ears(ear).IHC_coeffs.output_gain;
        
        CF.ears(ear).IHC_state.lpf1_state = CF.ears(ear).IHC_state.lpf1_state + CF.ears(ear).IHC_coeffs.lpf_coeff * ...
        (ihc_out - CF.ears(ear).IHC_state.lpf1_state);  % ʹ�õ�һ��˫��ƽ���˲�������ƽ��
    
        CF.ears(ear).IHC_state.lpf2_state = CF.ears(ear).IHC_state.lpf2_state + CF.ears(ear).IHC_coeffs.lpf_coeff * ...
        (CF.ears(ear).IHC_state.lpf1_state - CF.ears(ear).IHC_state.lpf2_state); % ʹ�õڶ���˫��ƽ���˲�������ƽ��
    
        ihc_out = CF.ears(ear).IHC_state.lpf2_state - CF.ears(ear).IHC_coeffs.rest_output;  % ���յõ����NAP
        
        CF.ears(ear).IHC_state.ihc_accum = CF.ears(ear).IHC_state.ihc_accum + ihc_out;  % for where decimated output is useful
        %  ��IHC��������ۼӣ���֮��ƽ�����NAP����
        
    %----------------------------------------------------------------------------------------------------
    %--------����AGC��״̬��ʵ��ͨ�������ϣ���DIHC�����NAP��Ϊ���룬���ص���CF.ears(ear).AGC_state-------
        % CF.ears(ear).AGC_state�е�AGC_memory�����P274ͼ19-6�����
        [CF.ears(ear).AGC_state, updated] = CARFAC_AGC_Step( ...
           ihc_out, CF.ears(ear).AGC_coeffs, CF.ears(ear).AGC_state);
       
        % save some output data:  
        seg_naps(k, :, ear) = ihc_out;  % ����DIHC�����NAP���ݣ�P266��ͼP18-7����
        % output to neural activity pattern ������񾭻ģʽ
        % �������naps = zeros(n_samp1, n_ch, n_ears);
        
        seg_BM(k, :, ear) = car_out;   %  ��DOHC�����Y����BM�У�P251��ͼ17-1��
        state = CF.ears(ear).CAR_state;  
        seg_ohc(k, :, ear) = state.zA_memory; % ��CAR_state�е�zA_memory����seg_ohc
        seg_agc(k, :, ear) = state.zB_memory;% ��CAR_state�е�zB_memory����seg_agc
        % �����BM = zeros(n_samp1, n_ch, n_ears);  % (֡���ȣ�71��n_ears)
        % seg_ohc = zeros(n_samp1, n_ch, n_ears);
        % seg_agc = zeros(n_samp1, n_ch, n_ears);
     end % for ear = 1:n_ears
     
    %--------------------------------------------------------------------------------------
    %----------------------------------ʵ��˫��������-------------------------------------
        if updated   % �����Ƿ�Ҫ�����
              % do multi-aural cross-coupling: �������������
              % CF.ears = CARFAC_Cross_Couple(CF.ears); %CF.ears������IHC��AGC��CAR����Ϣ
              % n_stages = ears(1).AGC_coeffs(1).n_AGC_stages;
              % now cross-ear mix the stages that updated (leading stages at phase 0):
              % ���ڽ�������ϸ��µĽ׶�(��0�׿�ʼ)
             n_stages=CF.ears(1).AGC_coeffs(1).n_AGC_stages;
             for stage = 1:n_stages
                if CF.ears(1).AGC_state(stage).decim_phase > 0
                  break  % ����������µĽ׶��Ѿ����
                else
                  mix_coeff = CF.ears(1).AGC_coeffs(stage).AGC_mix_coeffs;
                  if mix_coeff > 0  % Typically stage 1 has 0 so no work on that one.
                    this_stage_sum = 0;
                    % sum up over the ears and get their mean:
                    for ear = 1:n_ears
                      stage_state = CF.ears(ear).AGC_state(stage).AGC_memory;
                      % AGC_memory�����P274ͼ19-5���������4��ƽ���˲��������AGC_memory���е���
                      this_stage_sum = this_stage_sum + stage_state;
                    end
                    this_stage_mean = this_stage_sum / n_ears;
                    % now move them all toward the mean: �����Ƕ��Ƶ���ֵ��
                    for ear = 1:n_ears
                      stage_state = CF.ears(ear).AGC_state(stage).AGC_memory;
                      CF.ears(ear).AGC_state(stage).AGC_memory = ...
                        stage_state +  mix_coeff * (this_stage_mean - stage_state);
                    end
                  end
                end
             end
    %-------------------------------------------------------------------------------------
    %------------------------------�����Ƿ����AGC��Ԫʵ��---------------------------------
            if ~open_loop  % open_loop=0
              % CF = CARFAC_Close_AGC_Loop(CF);
              % ��AGC���b������CAR�Ը���CAR�Ĳ�����
              % ������CF.ears(ear).CAR_state.dzB_memory  drz
              % ������CF.ears(ear).CAR_state.dg_memory
              decim1 = CF.AGC_params.decimation(1);% decimation=[8,2,2,2]
            % ��AGC���b������CAR�Ը���CAR�Ĳ���
              for ear = 1:CF.n_ears
                  undamping = 1 - CF.ears(ear).AGC_state(1).AGC_memory; % stage 1 result
                  %��1-b��,b��ΪAGC�����AGC_memory
                  % Update the target stage gain for the new damping:
                  % ����Ŀ�꼶����Ϊ�µ�
                  new_r1 = CF.ears(ear).CAR_coeffs.r1_coeffs;  % at max damping
                  new_a0 = CF.ears(ear).CAR_coeffs.a0_coeffs;
                  new_c0 = CF.ears(ear).CAR_coeffs.c0_coeffs;
                  new_h  = CF.ears(ear).CAR_coeffs.h_coeffs;
                  new_zr = CF.ears(ear).CAR_coeffs.zr_coeffs;  %drz ��Ը�����
                  new_r  = r1 + zr .* undamping;  % r=r1+drz*(1-b)
                  new_g  = (1 - 2*new_r.*a0 + new_r.^2) ./ (1 - 2*new_r.*new_a0 + new_h.*new_r.*new_c0 + new_r.^2); 
                  % ������DC����õ�λ��������ļ�����g������P246��g�Ĺ�ʽ�ó�����ʾΪp251�е�����g��
                  % set the deltas needed to get to the new damping:
                  % �����µ����������delta(��)ֵ:
                  CF.ears(ear).CAR_state.dzB_memory = ...
                    (CF.ears(ear).CAR_coeffs.zr_coeffs .* undamping - ...
                    CF.ears(ear).CAR_state.zB_memory) / decim1;
                  CF.ears(ear).CAR_state.dg_memory = ...
                    (new_g - CF.ears(ear).CAR_state.g_memory) / decim1;
              end
            end
        end %(if updated)
    end  %for k=1:n_samp1
    
    %----------------------------------------------------------------------------
    %------------------------------��ÿ֡�������---------------------------------
    for ear = 1:n_ears
      % Accumulate segment BM to make full BM
      % �ۻ�BM
      BM(k_range, :, ear) = seg_BM(:, :, ear); % ������ȫ֡��DOHC+CAR���
      ohc(k_range, :, ear) = seg_ohc(:, :, ear);
      nap(k_range, :, ear) = seg_naps(:, :, ear);
      agc(k_range, :, ear) = seg_agc(:, :, ear);
      decim_naps(seg_num, :, ear) = CF.ears(ear).IHC_state.ihc_accum / seglen;
      CF.ears(ear).IHC_state.ihc_accum = zeros(n_ch,1); % ��ihc������ۼӳ�֡��
    end 
    
    %------------------------------��ÿ֡�������--------------------------------
    %------------------------��ʾAGC��ÿ��ƽ���˲��������-----------------------
%     figure(1);     
%     hold off;clf
%     maxmax = 0;
%     for ear = 1:n_ears 
%       hold on
%       for stage = 1:4;
%         stage_response = 2^(stage-1) * CF.ears(ear).AGC_state(stage).AGC_memory;
%         % ��ʾÿһ֡��ÿһ��ƽ���˲���AGC�����������ϼ�Ȩ����
%         plot(stage_response);
%         xlabel('ͨ����');
%         ylabel('AGC�˲���״̬�����b'); 
%         title('AGC�˲������ʵ�״̬');
%         maxmax = max(maxmax, max(stage_response));
%       end
%     end
%     axis([0, CF.n_ch+1, 0.0, maxmax * 1.01 + 0.002]);
%     drawnow   
end

%------------------------��ʾIHC�����NAPͼ-----------------------
 for ear = 1:n_ears
    smooth_nap = decim_naps(:, :, ear);
    mono_max = max(smooth_nap(:));
    if ear==1
        smooth_nap1=63 * ((max(0, smooth_nap)/mono_max)' .^ 0.5);
        imagesc(smooth_nap1);
        xlabel('ʱ��������');
        ylabel('CARFACͨ����'); 
        title('˫��ƽ��NAPͼ');
    end
%     else
%         smooth_nap2=63 * ((max(0, smooth_nap)/mono_max)' .^ 0.5);
%         imagesc(smooth_nap2);
%         xlabel('ʱ��������');
%         ylabel('CARFACͨ����'); 
%         title('˫��ƽ��NAPͼ');
%     end
%     if ear==1
%         save('C:\Users\PitayaFan\Desktop\nap_seg.txt','smooth_nap1','-ascii');
%     end
 end
 
 
  
%  for ear = 1:n_ears
%     naps = [];
%     naps = nap(:, :, ear);   % 240000*53
%     if ear==1
%         figure(3+ear);
%         nap1=naps';
%         imagesc(nap1);
%         xlabel('ʱ��������');
%         ylabel('CARFACͨ����'); 
%         title('δƽ��NAPͼ');
%     else
%         figure(3+ear);
%         nap2=naps';
%         imagesc(nap2);
%         xlabel('ʱ��������');
%         ylabel('CARFACͨ����'); 
%         title('˫��δƽ��NAPͼ');
%     end
%  end
%     figure(5+ear);
%     image(naps(10700:11000,:)'*10);
%     colormap(cmap);
%     xlabel('ʱ��������');
%     ylabel('CARFACͨ����'); 
%     title('˫��δƽ��NAPͼ(Ƭ��)');
%    naps_seg=naps(20000:20800,:);
%     if ear==1
%         save('C:\Users\Administrator\Desktop\nap2.txt','naps_seg','-ascii');
%     end

set(0,'defaultfigurecolor','w');


