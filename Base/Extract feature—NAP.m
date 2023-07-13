clear all
clc
close all
clear variables
%------------------------输入语音信号----------------------------------------
%---------------------------------------------------------------------------
wav='test_0_desc_30_300.wav';
[test_signal,fs]=audioread(wav);

% 直接考虑耳朵数量的情况
n_ears = 1;
%----------------------------------------------------------------------------
%---------------------------初始化CAR、IHC、AGC参数---------------------------
CF_CAR_params = struct( ...
    'velocity_scale', 0.1, ...  % P254中的非线性函数速度scale
    'v_offset', 0.04, ...  % P254中的offset
    'min_zeta', 0.10, ... % 最小阻尼因子ζ
    'max_zeta', 0.35, ... % 最大阻尼因子ζ
    'first_pole_theta', 0.85*pi, ... % 第一个极点角度的θ
    'zero_ratio', sqrt(2), ... % 零极点距离
    'high_f_damping_compression', 0.5, ... % 0 to 1 to compress zeta 高频阻尼压缩
    'ERB_per_step', 0.5, ... 
    'min_pole_Hz', 30, ...
    'ERB_break_freq', 165.3, ...  
    'ERB_Q', 1000/(24.7*4.37));  

just_hwr=0;
one_cap=1;
CF_IHC_params = struct( ...
    'just_hwr', just_hwr, ...        
    'one_cap', one_cap, ...   
    'tau_lpf', 0.000080, ...  % P266用于双极平滑滤波器
    'tau_out', 0.0005, ...    % P266中第一个低通滤波器的时间常数，通过减去输入信号以构成高通滤波器,用于抑制在基底膜波传播平方畸变时产生20hz以下的频率
    'tau_in', 0.010, ...      % P266用作IHC自动增益控制的环路滤波器的时间常数
    'ac_corner_Hz', 20);  % 抑制在基底膜波传播平方畸变时产生20hz以下的频率

CF_AGC_params = struct( ...
    'n_stages', 4, ...  % P269定义4阶的平滑滤波器
    'time_constants', 0.002 * 4.^(0:3), ... % 时间常量0.002,0.008,0.032,0.128
    'AGC_stage_gain', 2, ...  % 下一级更慢级的输入权重2
    'decimation', [8, 2, 2, 2], ...  % AGC更新时间点，叠乘后为[8,16,32,64]
    'AGC1_scales', 1.0 * sqrt(2).^(0:3), ...   %  蜗底向蜗顶平滑参数
    'AGC2_scales', 1.65 * sqrt(2).^(0:3), ... %  蜗顶向蜗底平滑参数
    'AGC_mix_coeff', 0.5);  %最大增益系数
%----------------------------------------------------------------------------
%-------------------------求通道数以及对应的中心频率---------------------------
pole_Hz = CF_CAR_params.first_pole_theta * fs / (2*pi);  % 通过第一极点的角度θ求得极点的频率
n_ch = 0;
while pole_Hz > CF_CAR_params.min_pole_Hz
  n_ch = n_ch + 1;
  pole_Hz = pole_Hz - CF_CAR_params.ERB_per_step * ...
    (CF_CAR_params.ERB_break_freq + pole_Hz) / CF_CAR_params.ERB_Q;
end
% ERB=24.7*(1+4.37*pole_Hz/1000),ERB_per_step=0.5,ERB_break_freq=165.3,ERB_Q=1000/(24.7*4.37)
% 求得通道数n_ch=71

% 求每个通道的中心频率pole_freqs从大到小排序的:(71,1)
pole_freqs = zeros(n_ch, 1);
pole_Hz = CF_CAR_params.first_pole_theta * fs / (2*pi);
for ch = 1:n_ch
  pole_freqs(ch) = pole_Hz;
  pole_Hz = pole_Hz - CF_CAR_params.ERB_per_step * ...
    (CF_CAR_params.ERB_break_freq + pole_Hz) / CF_CAR_params.ERB_Q;
end
max_channels_per_octave = log(2) / log(pole_freqs(1)/pole_freqs(2));% 求最大频道八度:12.2709
%----------------------------------------------------------------------------
%-----------------------------CAR_coeffs-------------------------------------
CAR_coeffs = struct( ...
  'n_ch', n_ch, ...   % 71
  'velocity_scale', CF_CAR_params.velocity_scale, ...  % P254 求NLF中的scale=0.1
  'v_offset', CF_CAR_params.v_offset ...   % P254 求NLF中的offset=0.04
  );
CAR_coeffs.r1_coeffs = zeros(n_ch, 1); % 极点半径，与阻尼因子有关
CAR_coeffs.a0_coeffs = zeros(n_ch, 1);%  a0为极点角的余弦
CAR_coeffs.c0_coeffs = zeros(n_ch, 1); % c0为极点角的正弦
CAR_coeffs.h_coeffs = zeros(n_ch, 1);  % h用于控制零点与极点的频率比率
CAR_coeffs.g0_coeffs = zeros(n_ch, 1); % g用于调整总增益
f = CF_CAR_params.zero_ratio^2 - 1; 
% zero_ratio表示为零极点距离sqrt(2),f的目的是为了求出h值，即h=c0*f
% P245,当零点频率将比极点频率高出大约半个倍频程时，f=1，h=c0
theta = pole_freqs .* (2 * pi / fs);
x = theta/pi; 
% 求得每个中心频率所对应的极点角度theta=θ，并将其转换为弧度x=w
c0 = sin(theta); % c0为极点角的正弦
a0 = cos(theta); % a0为极点角的余弦
ff = CF_CAR_params.high_f_damping_compression; % 可设置为0 to 1，此处定的0.5（高频阻尼压缩）
zr_coeffs = pi * (x - ff * x.^3); 
% when ff is 0, this is just theta, and when ff is 1 it goes to zero at theta = pi.
max_zeta = CF_CAR_params.max_zeta;   
% 'max_zeta', 0.35, 最大阻尼因子ζ
CAR_coeffs.r1_coeffs = (1 - zr_coeffs .* max_zeta);  
% r1是在最大阻尼条件下的半径
min_zeta = CF_CAR_params.min_zeta;
% 'min_zeta', 0.10，最小阻尼因子ζ
min_zetas = min_zeta + 0.25* ...
  (((CF_CAR_params.ERB_break_freq + pole_Hz) / CF_CAR_params.ERB_Q)./ pole_freqs - min_zeta);
% 根据等效矩阵带宽求出通道间距间更多的最小阻尼
CAR_coeffs.zr_coeffs = zr_coeffs .* ...
  (max_zeta - min_zetas);
% 求得相对负阻尼drz
CAR_coeffs.a0_coeffs = a0;
CAR_coeffs.c0_coeffs = c0;
h = c0 .* f;  
CAR_coeffs.h_coeffs = h;
relative_undamping = ones(n_ch, 1);  % 此处因为未引入AGC单元，假设b=0，故r=r1+drz*(1-b)中的(1-b)令为1
% CAR_coeffs.g0_coeffs = CARFAC_Stage_g(CAR_coeffs, relative_undamping);
r1 = CAR_coeffs.r1_coeffs;  % r1是在最大阻尼条件下的半径
a0 = CAR_coeffs.a0_coeffs;  
c0 = CAR_coeffs.c0_coeffs;
h  = CAR_coeffs.h_coeffs;  % h用于控制零点与极点的频率比率
zr = CAR_coeffs.zr_coeffs;  %drz 相对负阻尼
r  = r1 + zr .* relative_undamping;  % r=r1+drz*(1-b)
g  = (1 - 2*r.*a0 + r.^2) ./ (1 - 2*r.*a0 + h.*r.*c0 + r.^2); % 根据P246求g的公式得出

% AGC_coeffs = CARFAC_DesignAGC(AGC_params, fs, n_ch)
%----------------------------------------------------------------------------
%-----------------------------AGC_coeffs-------------------------------------
n_AGC_stages = CF_AGC_params.n_stages; % 平滑滤波器阶数4
% 'AGC1_scales', 1.0 * sqrt(2).^(0:3), ...   % AGC1通道从蜗底向蜗顶平滑
% 'AGC2_scales', 1.65 * sqrt(2).^(0:3), ...  % AGC2通道从蜗顶向蜗底平滑
AGC1_scales = CF_AGC_params.AGC1_scales;
AGC2_scales = CF_AGC_params.AGC2_scales;
decim = 1;
total_DC_gain = 0;
AGC_coeffs = struct([]);
for stage = 1:n_AGC_stages
  AGC_coeffs(stage).n_ch = n_ch;
  AGC_coeffs(stage).n_AGC_stages = n_AGC_stages;
  AGC_coeffs(stage).AGC_stage_gain = CF_AGC_params.AGC_stage_gain;
  % 'time_constants', 0.002 * 4.^(0:3), ... % 时间常量0.002,0.008,0.032,0.128
  % 'AGC_stage_gain', 2, ...  % 下一级更慢级的输入权重2
  % 'decimation', [8, 2, 2, 2], ...  % 根据叠乘更新时间点为[8,16,32,64]
  % 'AGC1_scales', 1.0 * sqrt(2).^(0:3)  AGC1通道从蜗底向蜗顶平滑
  % 'AGC2_scales', 1.65 * sqrt(2).^(0:3) AGC2通道从蜗顶向蜗底平滑
  AGC_coeffs(stage).decimation = CF_AGC_params.decimation(stage);
  tau = CF_AGC_params.time_constants(stage);  % 时间常量(s)
  decim = decim * CF_AGC_params.decimation(stage);  % 更新时间点，会根据阶数累乘 [8,16,32,64]
  % AGC_epsilon表示每一步更新需要多少新输入，理解：根据更新时间点、时间常数与采样率算出需要把前后多长时间的输入用来平滑
  AGC_coeffs(stage).AGC_epsilon = 1 - exp(-decim / (tau * fs)); 
  % 一个时间常数中平滑的有效次数:
  %（在每个AGC更新时间点作用于AGC低通滤波器状态数列，并利用时间平滑的长时常数进行多次有效迭代）
  ntimes = tau * (fs / decim);  % decim[8,16,32,64]

  % 确定脉冲响应的目标扩散(方差)和延迟(均值)作为要卷积n次的分布：
  delay = (AGC2_scales(stage) - AGC1_scales(stage)) / ntimes; 
  spread_sq = (AGC1_scales(stage)^2 + AGC2_scales(stage)^2) / ntimes; 
  % 获得极点位置，以更好地匹配每个方向上[[几何分布]]的预期扩展和延迟
  u = 1 + 1 / spread_sq;  
  p = u - sqrt(u^2 - 1);   
  dp = delay * (1 - 2*p +p^2)/2;
  polez1 = p - dp;
  polez2 = p + dp;
  AGC_coeffs(stage).AGC_polez1 = polez1;
  AGC_coeffs(stage).AGC_polez2 = polez2;

  % try a 3- or 5-tap FIR as an alternative to the double exponential:
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
        % 3点FIR:
        n_taps = 3;
      case 3
        % 5点FIR
        n_taps = 5;
      case 5
        % apply FIR multiple times instead of going wider:
        n_iterations = n_iterations + 1;
        if n_iterations > 4
          n_iteration = -1;  % Signal to use IIR instead.
        end
      otherwise
        % to do other n_taps would need changes in CARFAC_Spatial_Smooth
        % and in Design_FIR_coeffs
        error('Bad n_taps in CARFAC_DesignAGC');
    end
    % [AGC_spatial_FIR, done] = Design_FIR_coeffs(n_taps, spread_sq, delay, n_iterations);
    % 获得左右通道处的权重，返回值为AGC_spatial_FIR=[CL,1-CL-CR,CR]
    % 通过n次迭代减小平滑分布的均值和方差:
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
  AGC_coeffs(stage).AGC_spatial_iterations = n_iterations; % 空间迭代次数
  AGC_coeffs(stage).AGC_spatial_FIR = AGC_spatial_FIR;% 获得三侧权重
  AGC_coeffs(stage).AGC_spatial_n_taps = n_taps; % n_taps点FIR滤波器

  % 累积所有阶段的直流增益，占阶段增益：
  total_DC_gain = total_DC_gain + CF_AGC_params.AGC_stage_gain^(stage-1);


  if stage == 1
    AGC_coeffs(stage).AGC_mix_coeffs = 0;
  else
    AGC_coeffs(stage).AGC_mix_coeffs = CF_AGC_params.AGC_mix_coeff / ...
      (tau * (fs / decim));  % tau * (fs / decim)=ntimes 一个时间常数中平滑的有效次数
  end
end

% 将第1级detect_scale调整为AGC滤波器直流增益的倒数：
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
% CARFAC_Detect函数是将P266图18-7中的g算出来 
% ro = 1/CARFAC_Detect(10);
c = CF_IHC_params.tau_out / ro; % tau_out为0.5ms的放电（输出）时间常数 
ri = CF_IHC_params.tau_in / c;  % tau_in为10ms存储充电（输入）时间常数，为自动增益控制中LPF时间常数
saturation_output = 1 / (2*ro + ri); % 饱和下的输出

% 再考虑没有信号输入时数字IHC模型的平衡:
x_in2=0;
set2 = x_in2 > -a;
zz2 = x_in2(set2)+a;
conductance2 = zeros(size(x_in2));
conductance2(set2) = zz2.^3 ./ (zz2.^3 + zz2.^2 + 0.1);  
r0 = 1 / conductance2; 
% r0 = 1/CARFAC_Detect(0);
current = 1 / (ri + r0);  % 直流q
cap_voltage = 1 - current * ri;  % 上限电压v
IHC_coeffs = struct( ...
  'n_ch', n_ch, ...
  'just_hwr', 0, ...
  'lpf_coeff', 1 - exp(-1/(CF_IHC_params.tau_lpf * fs)), ...
  'out_rate', ro / (CF_IHC_params.tau_out * fs), ...
  'in_rate', 1 / (CF_IHC_params.tau_in * fs), ...
  'one_cap', CF_IHC_params.one_cap, ...
  'output_gain', 1/ (saturation_output - current), ... % 归一化
  'rest_output', current / (saturation_output - current), ...
  'rest_cap', cap_voltage);
% 测试/验证的单通道状态:
IHC_state = struct( ...
  'cap_voltage', IHC_coeffs.rest_cap, ...
  'lpf1_state', 0, ...
  'lpf2_state', 0, ...
  'ihc_accum', 0);
IHC_coeffs.ac_coeff = 2 * pi * CF_IHC_params.ac_corner_Hz / fs;

%-----------------------------------------------------------------------------
%----------------创建CF结构体将上边所得到的结构体和参数存入进去------------------
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
%---------------初始化各部件运行的临时状态参数  CF = CARFAC_Init(CF);----------------
for ear = 1:n_ears
    n_ch = CF.ears(ear).CAR_coeffs.n_ch;
    CF.ears(ear).CAR_state = struct(...
      'z1_memory', zeros(n_ch, 1), ...
      'z2_memory', zeros(n_ch, 1), ...
      'zA_memory', zeros(n_ch, 1), ...
      'zB_memory', CF.ears(ear).CAR_coeffs.zr_coeffs, ...% dzr:CAR_coeffs.zr_coeffs = zr_coeffs .*(max_zeta - min_zetas);
      'dzB_memory', zeros(n_ch, 1), ...
      'zY_memory', zeros(n_ch, 1), ...
      'g_memory', CF.ears(ear).CAR_coeffs.g0_coeffs, ...% p251图17-1中的增益g
      'dg_memory', zeros(n_ch, 1) ...
      );
    % CAR_Init_State后还留有z1_memory，z2_memory，zA_memory，zB_memory（非0），dzB_memory
    % zY_memory，g_memory（非0），dg_memory

    n_ch = CF.ears(ear).IHC_coeffs.n_ch;
    CF.ears(ear).IHC_state = struct(...
      'ihc_accum', zeros(n_ch, 1), ...
      'cap_voltage', CF.ears(ear).IHC_coeffs.rest_cap * ones(n_ch, 1), ...
      'lpf1_state', CF.ears(ear).IHC_coeffs.rest_output * ones(n_ch, 1), ...
      'lpf2_state', CF.ears(ear).IHC_coeffs.rest_output * ones(n_ch, 1), ...
      'ac_coupler', zeros(n_ch, 1) ...
      );

     n_ch = CF.ears(ear).AGC_coeffs(1).n_ch; % 取通道数
     n_AGC_stages = CF.ears(ear).AGC_coeffs.n_AGC_stages;% 取平滑滤波器的阶数
     CF.ears(ear).AGC_state = struct([]);
    for stage = 1:n_AGC_stages
         CF.ears(ear).AGC_state(stage).AGC_memory = zeros(n_ch, 1);
         CF.ears(ear).AGC_state(stage).input_accum = zeros(n_ch, 1);
         CF.ears(ear).AGC_state(stage).decim_phase = 0;  % integer decimator phase
    end
  % CF.ears(ear).IHC_coeffs
  % IHC_Init_State后还留有ihc_accum，cap_voltage（非0），lpf1_state（非0），lpf2_state（非0），ac_coupler
  % AGC_Init_State后还留有每级平滑滤波器的AGC_memory，input_accum，decim_phase=0
end
%----------------------------------------------------------------------------------
%-------------------------------------求帧数---------------------------------------
% agc_plot_fig_num = 10;
%[CF_struct, nap_decim, nap, BM, ohc, agc] = CARFAC_Run(CF_struct, test_signal,agc_plot_fig_num);
[n_samp,n_ears] = size(test_signal);
n_ch = CF.n_ch;
BM = zeros(n_samp,n_ch,n_ears);
ohc = zeros(n_samp,n_ch,n_ears);
agc = zeros(n_samp,n_ch,n_ears);
nap = zeros(n_samp,n_ch,n_ears);
seglen = 256; % 设帧长为160   882→20ms一帧(fs=44100) 882*
seg_move = 80;
step = seglen - seg_move;
n_segs = floor((n_samp-seg_move)/(seglen - seg_move))
decim_naps = zeros(n_segs,CF.n_ch,CF.n_ears);
if n_ears~=CF.n_ears
    error('bad number of input_waves channels passed to CARFAC_Run')
end
%----------------------------------------------------------------------------------
%--------------------------------------按帧运行------------------------------------
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
    % test_signal(k_range, :)为每一帧的数据
    open_loop = 0;
    % do_BM = 1;  因为要显示BM图，所以do_BM令为1
    input_waves = test_signal(k_range,:).*win;
    [n_samp1,n_ears] = size(input_waves);  % 此时n_samp为每一帧的长度
    if n_ears ~= CF.n_ears
        error('bad number of input_waves channels passed to CARFAC_Run')
    end
    seg_naps = zeros(n_samp1, n_ch, n_ears); 
    seg_BM = zeros(n_samp1, n_ch, n_ears);  % (信号长度，71，n_ears)
    seg_ohc = zeros(n_samp1, n_ch, n_ears);
    seg_agc = zeros(n_samp1, n_ch, n_ears);
    
    for k = 1:n_samp1
  % at each time step, possibly handle multiple channels  在每个时间步骤中，可能处理多个通道
      for ear = 1:n_ears
   %--------------------------------------------------------------------------------------
   %------------更新CAR的状态，car_out是返回P251中图17-1中DOHC的输出Y----------------------
        %  [car_out, CF.ears(ear).CAR_state] = CARFAC_CAR_Step( ...
        %   input_waves(k, ear), CF.ears(ear).CAR_coeffs, CF.ears(ear).CAR_state);
        g = CF.ears(ear).CAR_state.g_memory+CF.ears(ear).CAR_state.dg_memory;
        zB = CF.ears(ear).CAR_state.zB_memory + CF.ears(ear).CAR_state.dzB_memory;
        zA = CF.ears(ear).CAR_state.zA_memory;
        v = CF.ears(ear).CAR_state.z2_memory - zA;
        nlf = 1 ./ (1 + (v * CAR_coeffs.velocity_scale + CAR_coeffs.v_offset) .^ 2 ); % p254
        r = CAR_coeffs.r1_coeffs + zB .* nlf; % p255求极点半径，此处的r=r1+drz*NLF(还未乘AGC的输出（1-b）)
        zA = CF.ears(ear).CAR_state.z2_memory;
        z1 = r .* (CAR_coeffs.a0_coeffs .* ...  % z1 = z1 + inputs; 图17-1上通路
        CF.ears(ear).CAR_state.z1_memory - CAR_coeffs.c0_coeffs .* CF.ears(ear).CAR_state.z2_memory);
        z2 = r .* (CAR_coeffs.c0_coeffs .* ...  % 图17-1下通路
        CF.ears(ear).CAR_state.z1_memory + CAR_coeffs.a0_coeffs .* CF.ears(ear).CAR_state.z2_memory);
        zY = CAR_coeffs.h_coeffs .* z2;    % 部分输出h的那条通路
        in_out = input_waves(k,ear);
        for ch = 1:length(zY)
          z1(ch) = z1(ch) + in_out;
          in_out = g(ch) * (in_out + zY(ch));  % P251图17-1输出：Y=g*(x+h)
          zY(ch) = in_out;
        end
        CF.ears(ear).CAR_state.z1_memory = z1;
        CF.ears(ear).CAR_state.z2_memory = z2;
        CF.ears(ear).CAR_state.zA_memory = zA;  % 表示下通路的输出（需要乘h输出）
        CF.ears(ear).CAR_state.zB_memory = zB;  % drz
        CF.ears(ear).CAR_state.zY_memory = zY;
        CF.ears(ear).CAR_state.g_memory = g;
        car_out = zY;
        
    %--------------------------------------------------------------------------------------
    %------更新IHC的状态，将DOHC的输出Y作为输入，ihc_out是返回P266中图P18-7中DIHC的输出NAP----
        %   [ihc_out, CF.ears(ear).IHC_state] = CARFAC_IHC_Step( ...
        %   car_out, CF.ears(ear).IHC_coeffs, CF.ears(ear).IHC_state);
        ac_diff = car_out - CF.ears(ear).IHC_state.ac_coupler; 
        CF.ears(ear).IHC_state.ac_coupler = CF.ears(ear).IHC_state.ac_coupler + CF.ears(ear).IHC_coeffs.ac_coeff * ac_diff;
        % conductance = CARFAC_Detect(ac_diff); P266半波整流、NLF模块，conductance为g
        set = ac_diff > -a;
        zz = ac_diff(set)+a;
        conductance = zeros(size(ac_diff));
        conductance(set) = zz.^3 ./ (zz.^3 + zz.^2 + 0.1);
        ihc_out = conductance .* CF.ears(ear).IHC_state.cap_voltage;  % p266图18-7 y=g*v
        CF.ears(ear).IHC_state.cap_voltage = CF.ears(ear).IHC_state.cap_voltage - ihc_out .* CF.ears(ear).IHC_coeffs.out_rate + ...
        (1 - CF.ears(ear).IHC_state.cap_voltage) .* CF.ears(ear).IHC_coeffs.in_rate;
        ihc_out = ihc_out * CF.ears(ear).IHC_coeffs.output_gain;
        
        CF.ears(ear).IHC_state.lpf1_state = CF.ears(ear).IHC_state.lpf1_state + CF.ears(ear).IHC_coeffs.lpf_coeff * ...
        (ihc_out - CF.ears(ear).IHC_state.lpf1_state);  % 使用第一个双极平滑滤波器进行平滑
    
        CF.ears(ear).IHC_state.lpf2_state = CF.ears(ear).IHC_state.lpf2_state + CF.ears(ear).IHC_coeffs.lpf_coeff * ...
        (CF.ears(ear).IHC_state.lpf1_state - CF.ears(ear).IHC_state.lpf2_state); % 使用第二个双极平滑滤波器进行平滑
    
        ihc_out = CF.ears(ear).IHC_state.lpf2_state - CF.ears(ear).IHC_coeffs.rest_output;  % 最终得到输出NAP
        
        CF.ears(ear).IHC_state.ihc_accum = CF.ears(ear).IHC_state.ihc_accum + ihc_out;  % for where decimated output is useful
        %  对IHC输出进行累加，对之后平滑输出NAP有用
        
    %----------------------------------------------------------------------------------------------------
    %--------更新AGC的状态并实现通道间的耦合，将DIHC的输出NAP作为输入，返回的是CF.ears(ear).AGC_state-------
        % CF.ears(ear).AGC_state中的AGC_memory存的是P274图19-6的输出
        [CF.ears(ear).AGC_state, updated] = CARFAC_AGC_Step( ...
           ihc_out, CF.ears(ear).AGC_coeffs, CF.ears(ear).AGC_state);
       
        % save some output data:  
        seg_naps(k, :, ear) = ihc_out;  % 保存DIHC的输出NAP数据（P266中图P18-7）：
        % output to neural activity pattern 输出到神经活动模式
        % 定义的是naps = zeros(n_samp1, n_ch, n_ears);
        
        seg_BM(k, :, ear) = car_out;   %  将DOHC的输出Y存入BM中（P251中图17-1）
        state = CF.ears(ear).CAR_state;  
        seg_ohc(k, :, ear) = state.zA_memory; % 将CAR_state中的zA_memory赋予seg_ohc
        seg_agc(k, :, ear) = state.zB_memory;% 将CAR_state中的zB_memory赋予seg_agc
        % 定义的BM = zeros(n_samp1, n_ch, n_ears);  % (帧长度，71，n_ears)
        % seg_ohc = zeros(n_samp1, n_ch, n_ears);
        % seg_agc = zeros(n_samp1, n_ch, n_ears);
     end % for ear = 1:n_ears
     
    %--------------------------------------------------------------------------------------
    %----------------------------------实现双耳间的耦合-------------------------------------
        if updated   % 讨论是否要用耦合
              % do multi-aural cross-coupling: 多声道交叉耦合
              % CF.ears = CARFAC_Cross_Couple(CF.ears); %CF.ears包含了IHC、AGC、CAR的信息
              % n_stages = ears(1).AGC_coeffs(1).n_AGC_stages;
              % now cross-ear mix the stages that updated (leading stages at phase 0):
              % 现在交叉耳朵混合更新的阶段(在0阶开始)
             n_stages=CF.ears(1).AGC_coeffs(1).n_AGC_stages;
             for stage = 1:n_stages
                if CF.ears(1).AGC_state(stage).decim_phase > 0
                  break  % 所有最近更新的阶段已经完成
                else
                  mix_coeff = CF.ears(1).AGC_coeffs(stage).AGC_mix_coeffs;
                  if mix_coeff > 0  % Typically stage 1 has 0 so no work on that one.
                    this_stage_sum = 0;
                    % sum up over the ears and get their mean:
                    for ear = 1:n_ears
                      stage_state = CF.ears(ear).AGC_state(stage).AGC_memory;
                      % AGC_memory存的是P274图19-5的输出，对4个平滑滤波器的输出AGC_memory进行叠加
                      this_stage_sum = this_stage_sum + stage_state;
                    end
                    this_stage_mean = this_stage_sum / n_ears;
                    % now move them all toward the mean: 把它们都移到均值：
                    for ear = 1:n_ears
                      stage_state = CF.ears(ear).AGC_state(stage).AGC_memory;
                      CF.ears(ear).AGC_state(stage).AGC_memory = ...
                        stage_state +  mix_coeff * (this_stage_mean - stage_state);
                    end
                  end
                end
             end
    %-------------------------------------------------------------------------------------
    %------------------------------考虑是否添加AGC单元实现---------------------------------
            if ~open_loop  % open_loop=0
              % CF = CARFAC_Close_AGC_Loop(CF);
              % 将AGC输出b反馈给CAR以更新CAR的参数：
              % 更新了CF.ears(ear).CAR_state.dzB_memory  drz
              % 更新了CF.ears(ear).CAR_state.dg_memory
              decim1 = CF.AGC_params.decimation(1);% decimation=[8,2,2,2]
            % 将AGC输出b反馈给CAR以更新CAR的参数
              for ear = 1:CF.n_ears
                  undamping = 1 - CF.ears(ear).AGC_state(1).AGC_memory; % stage 1 result
                  %（1-b）,b即为AGC的输出AGC_memory
                  % Update the target stage gain for the new damping:
                  % 更新目标级增益为新的
                  new_r1 = CF.ears(ear).CAR_coeffs.r1_coeffs;  % at max damping
                  new_a0 = CF.ears(ear).CAR_coeffs.a0_coeffs;
                  new_c0 = CF.ears(ear).CAR_coeffs.c0_coeffs;
                  new_h  = CF.ears(ear).CAR_coeffs.h_coeffs;
                  new_zr = CF.ears(ear).CAR_coeffs.zr_coeffs;  %drz 相对负阻尼
                  new_r  = r1 + zr .* undamping;  % r=r1+drz*(1-b)
                  new_g  = (1 - 2*new_r.*a0 + new_r.^2) ./ (1 - 2*new_r.*new_a0 + new_h.*new_r.*new_c0 + new_r.^2); 
                  % 返回在DC处获得单位增益所需的级增益g（根据P246求g的公式得出，显示为p251中的增益g）
                  % set the deltas needed to get to the new damping:
                  % 设置新的阻尼所需的delta(δ)值:
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
    %------------------------------将每帧组合起来---------------------------------
    for ear = 1:n_ears
      % Accumulate segment BM to make full BM
      % 累积BM
      BM(k_range, :, ear) = seg_BM(:, :, ear); % 求语音全帧的DOHC+CAR输出
      ohc(k_range, :, ear) = seg_ohc(:, :, ear);
      nap(k_range, :, ear) = seg_naps(:, :, ear);
      agc(k_range, :, ear) = seg_agc(:, :, ear);
      decim_naps(seg_num, :, ear) = CF.ears(ear).IHC_state.ihc_accum / seglen;
      CF.ears(ear).IHC_state.ihc_accum = zeros(n_ch,1); % 用ihc的输出累加除帧长
    end 
    
    %------------------------------将每帧组合起来--------------------------------
    %------------------------显示AGC中每个平滑滤波器的输出-----------------------
%     figure(1);     
%     hold off;clf
%     maxmax = 0;
%     for ear = 1:n_ears 
%       hold on
%       for stage = 1:4;
%         stage_response = 2^(stage-1) * CF.ears(ear).AGC_state(stage).AGC_memory;
%         % 显示每一帧、每一级平滑滤波器AGC输出的增益乘上加权因子
%         plot(stage_response);
%         xlabel('通道号');
%         ylabel('AGC滤波器状态及输出b'); 
%         title('AGC滤波器的适调状态');
%         maxmax = max(maxmax, max(stage_response));
%       end
%     end
%     axis([0, CF.n_ch+1, 0.0, maxmax * 1.01 + 0.002]);
%     drawnow   
end

%------------------------显示IHC输出的NAP图-----------------------
 for ear = 1:n_ears
    smooth_nap = decim_naps(:, :, ear);
    mono_max = max(smooth_nap(:));
    if ear==1
        smooth_nap1=63 * ((max(0, smooth_nap)/mono_max)' .^ 0.5);
        imagesc(smooth_nap1);
        xlabel('时间样点数');
        ylabel('CARFAC通道数'); 
        title('双耳平滑NAP图');
    end
%     else
%         smooth_nap2=63 * ((max(0, smooth_nap)/mono_max)' .^ 0.5);
%         imagesc(smooth_nap2);
%         xlabel('时间样点数');
%         ylabel('CARFAC通道数'); 
%         title('双耳平滑NAP图');
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
%         xlabel('时间样点数');
%         ylabel('CARFAC通道数'); 
%         title('未平滑NAP图');
%     else
%         figure(3+ear);
%         nap2=naps';
%         imagesc(nap2);
%         xlabel('时间样点数');
%         ylabel('CARFAC通道数'); 
%         title('双耳未平滑NAP图');
%     end
%  end
%     figure(5+ear);
%     image(naps(10700:11000,:)'*10);
%     colormap(cmap);
%     xlabel('时间样点数');
%     ylabel('CARFAC通道数'); 
%     title('双耳未平滑NAP图(片段)');
%    naps_seg=naps(20000:20800,:);
%     if ear==1
%         save('C:\Users\Administrator\Desktop\nap2.txt','naps_seg','-ascii');
%     end

set(0,'defaultfigurecolor','w');


