function [gbest, gbestval, nfe, RecordT] = QIDE_func(fhd, D, nfe_max, Bd, varargin)
% QIDE(多中心聚类变异)
%

%% ===== 参数解析 =====
seed = 123;
stm = RandStream('swb2712','Seed',seed);
RandStream.setGlobalStream(stm);

fidvec = cell2mat(varargin);
fid = fidvec(1);
runid = fidvec(2);
Xmax = Bd(1, :);
Xmin = Bd(2, :);

%% ===== 算法参数 =====
% --- 基础参数（与HXDE一致） ---
H = 6;                          % 记忆池大小
arc_rate = 2.6;                 % 外部档案比率
p_rate = 0.11;                  % pbest精英比例
NP0 = max(30, round(18 * D));   % 初始种群规模
NPmin = 4;                      % 最小种群规模
ST = 30;                        % 学习向量停滞阈值
tau = 0.4;                      % 学习交叉触发概率
lv_cr_boost = 0.25;             % 学习交叉CR增强
dls_min_evidence = 2;           % 单维方向生效所需的最少观测数
dls_neg_repeat = 3;             % 负方向达到完整作用所需的重复失败数
dls_lambda_max = 0.5;           % 负方向信息的最大作用强度
ge_switch = 0.35;               % GE多样性阶段切换阈值 (V3: 0.3→0.35)
phase1_ratio = 0.6;             % Phase1基础占比

% --- Q-learning参数 (V3: 2状态→4状态) ---
alpha_ql = 0.2;                 % 学习率
% alpha_ql = 0.1;                 % 学习率
gamma_ql = 0.8;                 % 折扣因子
epsilon = 0.15;                 % ε-greedy探索率
Q_table = zeros(4, 3);          % 4状态 × 3动作 (V3改进)
stall_gen = 0;                  % 全局停滞计数器

% --- 基因混洗参数 (V3新增，替代指数交叉) ---
n_shuffle_parents = 5;          % 混洗亲本数量

% --- 个体级重启参数 (V3改进) ---
restart_stag_lim = max(20, round(0.08 * nfe_max / NP0));  % 个体停滞重启阈值(代数)
restart_max_per_gen_frac = 0.05;  % 每代最大重启比例

%% ===== CEC2017目标最优值 (30维安全数组) =====
targetbest = zeros(1, 30);
targetbest(1) = 100;
for i = 2:30
    targetbest(i) = i * 100;
end

%% ===== 日志文件 =====
logFile = ['QHXDE_v3_MC_fid_', num2str(fid), '_', ...
    num2str(D), 'D_', num2str(runid), '.dat'];
fout = fopen(logFile, 'wt');
if fout == -1
    error('QHXDE_v3_MC:LogOpenFailed', '无法创建日志文件：%s', logFile);
end
cleanupFile = onCleanup(@() fclose(fout)); %#ok<NASGU>

Vbnd = sqrt(max(prod(abs(Xmax - Xmin)), realmin));

%% ===== SHADE记忆初始化 =====
MF = 0.5 * ones(H, 1);
MCR = 0.5 * ones(H, 1);
MSigmaF = 0.1 * ones(H, 1);
k = 1;

%% ===== 初始化种群 =====
NP = NP0;
pop = Xmin + (Xmax - Xmin) .* rand(NP, D);
fx = feval(fhd, pop', fid);
fx = fx(:);
nfe = NP;
[gbestval, bi] = min(fx);
gbest = pop(bi, :);

fprintf(fout, '%d\t%.15f\n', nfe, gbestval - targetbest(fid));

A = zeros(0, D);                % 外部档案
print_interval = nfe_max / 20;
print_num = print_interval;
last_imp_nfe = nfe;
S_sigmaF_hist = [];

%% ===== 改进DLS的正、负方向记忆初始化 =====
LV_pos = zeros(NP, D);          % 有利试验位移的逐维累积和
LF_pos = zeros(NP, D);          % 有利试验位移的逐维观测次数
LV_neg = zeros(NP, D);          % 不利试验位移的逐维累积和
LF_neg = zeros(NP, D);          % 不利试验位移的逐维观测次数
cts = zeros(NP, 1);
Yfit = fx;

iter = 1;

tic;
%% ==================== 主循环 ====================
while nfe < nfe_max
    iter = iter + 1;
    
    %% --- 排序 + GE多样性计算 ---
    [~, ord] = sort(fx, 'ascend');
    pNP = max(2, round(p_rate * NP));
    
    %% --- V3_MC新增：多中心聚类分组 ---
    % 将种群按适应度排名交错分为K个簇，每个簇包含好/中/差个体
    % 各簇独立搜索不同区域，增强多峰问题的覆盖能力
    n_centers = 3;
    cluster_id = zeros(NP, 1);
    for ci = 1:NP
        % ord(ci)是排名为ci的个体编号，因此必须把簇标签写回该个体。
        % 原写法cluster_id(ci)=mod(ord(ci)-1,...)混淆了“排名”和
        % “个体编号”，不能保证各簇按适应度排名交错分组。
        cluster_id(ord(ci)) = mod(ci - 1, n_centers) + 1;
    end
    
    % GE (Gene Expression diversity) 计算
    ux = max(pop, [], 1);
    lx = min(pop, [], 1);
    Vdist = sqrt(max(prod(abs((ux - lx) / 2)), realmin));
    GE = sqrt(Vdist / Vbnd);
    prog = nfe / nfe_max;
    
    %% --- 阶段判断 (V3微调) ---
    if GE < ge_switch
        current_phase = 1;      % 种群聚集 → Wavelet探索
    else
        current_phase = 2;      % 种群分散 → Cauchy开发
    end
    if prog < 0.05
        current_phase = 1;      % 初始强制探索
    end
    if prog > 0.85
        current_phase = 2;      % 末期强制开发
    end
    
    %% --- V3改进：4状态Q-learning ---
    % 状态编码: (多样性水平, 停滞状态) → 4个语义明确的状态
    diversity_high = (GE > ge_switch);   % 1=种群分散, 0=种群聚集
    stall_active = (stall_gen > 3);      % 1=停滞, 0=正常
    
    curr_state = diversity_high * 2 + stall_active + 1;
    % 状态1: 聚集+正常 → 开发正常收敛中
    % 状态2: 聚集+停滞 → 陷入局部最优需逃逸
    % 状态3: 分散+正常 → 探索正常进行中
    % 状态4: 分散+停滞 → 探索方向不对需调整
    
    if rand < epsilon
        curr_action = randi(3);
    else
        [~, curr_action] = max(Q_table(curr_state, :));
    end
    
    old_gbestval = gbestval;
    old_avgfx = mean(fx);
    
    %% --- 从记忆池采样F/CR/σF参数 ---
    MR = randi(H, NP, 1);
    muF = MF(MR);
    muCR = MCR(MR);
    muSigmaF = MSigmaF(MR);
    
    %% --- F值生成（两阶段：Wavelet/Cauchy） ---
    F = zeros(NP, 1);
    ColSigmaF = zeros(NP, 1);
    for i = 1:NP
        sigma_F = muSigmaF(i);
        if ~isempty(S_sigmaF_hist) && numel(S_sigmaF_hist) > 5
            sigmaF_std = std(S_sigmaF_hist);
            if sigmaF_std > 0.05
                sigma_F = sigma_F * 1.2;
            else
                sigma_F = sigma_F * 0.8;
            end
        end
        sigma_F = min(max(sigma_F, 0.01), 0.2);
        ColSigmaF(i) = sigma_F;
        
        if current_phase == 1
            F(i) = randWavelet(muF(i), sigma_F, prog);
            guard = 0;
            while (F(i) <= 0 || F(i) > 1.0) && guard < 100
                F(i) = randWavelet(muF(i), sigma_F, prog);
                guard = guard + 1;
            end
            F(i) = min(max(F(i), 1e-8), 1.0);
        else
            F(i) = randCauchy(muF(i), sigma_F * 0.5);
            guard = 0;
            while (F(i) <= 0 || F(i) > 0.8) && guard < 100
                F(i) = randCauchy(muF(i), sigma_F * 0.5);
                guard = guard + 1;
            end
            F(i) = min(max(F(i), 1e-8), 0.8);
        end
    end
    
    %% --- CR生成 ---
    CR = muCR + 0.1 * randn(NP, 1);
    stag_ratio = min(cts / ST, 1);
    CR = CR + 0.15 * stag_ratio;
    CR(MCR(MR) == -1) = 0;
    CR = min(max(CR, 0), 1);
    
    %% --- 多中心pbest池 + 随机索引预生成 ---
    % 每个个体从自己所属簇内选pbest（不同簇探索不同区域）
    Xpb = zeros(NP, D);
    for c = 1:n_centers
        c_mask = (cluster_id == c);
        c_np = sum(c_mask);
        local_pNP = min(pNP, c_np);
        [~, c_ord] = sort(fx(c_mask), 'ascend');
        c_indices = find(c_mask);
        local_pbest_idx = c_indices(c_ord(1:local_pNP));
        for i = find(c_mask)'
            Xpb(i, :) = pop(local_pbest_idx(randi(local_pNP)), :);
        end
    end
    NA = size(A, 1);
    popA = [pop; A];
    Ltot = NP + NA;
    
    r1 = randi(NP, NP, 1);
    for i = 1:NP
        while r1(i) == i, r1(i) = randi(NP); end
    end
    r2 = randi(Ltot, NP, 1);
    for i = 1:NP
        while r2(i) == i || (r2(i) <= NP && r2(i) == r1(i))
            r2(i) = randi(Ltot);
        end
    end
    
    %% ===== 变异 + 交叉 =====
    U = zeros(NP, D);
    CrMask = false(NP, D);
    Delta = zeros(NP, D);
    
    switch curr_action
        case 1
            % ---------------------------------------------------------
            % Action 1: current-to-pbest/1 + 二项交叉 (主力策略)
            % 朝top精英方向移动 + 随机差分扰动
            % ---------------------------------------------------------
            for i = 1:NP
                v = pop(i,:) + F(i)*(Xpb(i,:) - pop(i,:)) ...
                    + F(i)*(pop(r1(i),:) - popA(r2(i),:));
                [ui, mi] = lshade_cross_mask(pop(i,:), v, CR(i), D);
                U(i,:) = ui;
                CrMask(i,:) = mi;
                Delta(i, mi) = v(mi) - pop(i, mi);
            end
            
        case 2
            % ---------------------------------------------------------
            % Action 2: 基因混洗 - V3核心改进 (替代指数交叉)
            % 从5个优质亲本中逐维度随机选取基因拼接，
            % 再加F扰动+binomial交叉，组合空间远大于2亲本混合
            % ---------------------------------------------------------
            % n_parents = min(n_shuffle_parents, pNP);
            % for i = 1:NP
            %     % 从top个体中随机选n_parents个亲本
            %     parent_sel = ord(randperm(pNP, n_parents));
            % 
            %     % 随机分块基因混洗：尽量保留变量之间的协同关系
            %     x_shuffle = pop(parent_sel(1), :);
            % 
            %     block_size = max(2, round(sqrt(D)));
            %     dim_order = randperm(D);
            % 
            %     for k_block = 1:block_size:D
            %         block_dims = dim_order(k_block:min(k_block + block_size - 1, D));
            %         p = parent_sel(randi(n_parents));
            %         x_shuffle(block_dims) = pop(p, block_dims);
            %     end
            % 
            %     % F扰动：朝pbest方向 + 随机差分
            %     v = x_shuffle + F(i)*(Xpb(i,:) - x_shuffle) ...
            %         + F(i)*(pop(r1(i),:) - popA(r2(i),:));
            % 
            %     % Binomial交叉融合
            %     [ui, mi] = lshade_cross_mask(pop(i,:), v, CR(i), D);
            %     U(i,:) = ui;
            %     CrMask(i,:) = mi;
            %     Delta(i, mi) = v(mi) - pop(i, mi);
            % end

            % for i = 1:NP
            % 
            %     % 实际使用的亲本数量
            %     n_parents = min(n_shuffle_parents, NP);
            % 
            %     % 约一半来自精英池，其余来自非精英池
            %     n_elite = min(pNP, ceil(n_parents / 2));
            %     n_normal = n_parents - n_elite;
            % 
            %     % 选择精英亲本
            %     elite_pool = ord(1:pNP);
            %     elite_sel = elite_pool(randperm(numel(elite_pool), n_elite));
            % 
            %     % 选择普通亲本
            %     normal_pool = ord(pNP+1:end);
            % 
            %     if n_normal > 0 && ~isempty(normal_pool)
            %         n_normal = min(n_normal, numel(normal_pool));
            %         normal_sel = normal_pool( ...
            %             randperm(numel(normal_pool), n_normal));
            %     else
            %         normal_sel = [];
            %     end
            % 
            %     parent_sel = [elite_sel(:); normal_sel(:)];
            % 
            %     % 极小种群情况下，确保至少存在一个亲本
            %     if isempty(parent_sel)
            %         parent_sel = ord(1);
            %     end
            % 
            %     % 随机打乱亲本顺序，避免第一个亲本总是精英
            %     parent_sel = parent_sel(randperm(numel(parent_sel)));
            % 
            %     % -----------------------------------------------------
            %     % 随机分块基因混洗
            %     % -----------------------------------------------------
            %     x_shuffle = pop(parent_sel(1), :);
            % 
            %     block_size = max(2, round(sqrt(D)));
            %     dim_order = randperm(D);
            % 
            %     for k_block = 1:block_size:D
            %         block_dims = dim_order( ...
            %             k_block:min(k_block + block_size - 1, D));
            % 
            %         p = parent_sel(randi(numel(parent_sel)));
            %         x_shuffle(block_dims) = pop(p, block_dims);
            %     end
            % 
            %     % pbest引导 + 差分扰动
            %     v = x_shuffle ...
            %         + F(i) * (Xpb(i,:) - x_shuffle) ...
            %         + F(i) * (pop(r1(i),:) - popA(r2(i),:));
            % 
            %     % 与当前个体进行二项式交叉
            %     [ui, mi] = lshade_cross_mask( ...
            %         pop(i,:), v, CR(i), D);
            % 
            %     U(i,:) = ui;
            %     CrMask(i,:) = mi;
            %     Delta(i,mi) = v(mi) - pop(i,mi);
            % end
            
           for i = 1:NP
        
                % 当前个体所属簇
                cid = cluster_id(i);
        
                % 簇内和簇外候选池
                local_pool = find(cluster_id == cid);
                external_pool = find(cluster_id ~= cid);
        
                % 不让当前个体作为自己的混洗亲本
                local_pool(local_pool == i) = [];
        
                % 总亲本数量
                n_parents = min(n_shuffle_parents, NP - 1);
                n_parents = max(1, n_parents);
        
                % 70%簇内、30%簇外
                n_local = round(0.4 * n_parents);
                n_external = n_parents - n_local;
        
                % 根据候选池大小调整数量
                n_local = min(n_local, numel(local_pool));
                n_external = min(n_external, numel(external_pool));
        
                % 如果簇内数量不足，优先从簇外补足
                missing = n_parents - n_local - n_external;
        
                if missing > 0
                    external_available = ...
                        numel(external_pool) - n_external;
        
                    add_external = min(missing, external_available);
                    n_external = n_external + add_external;
                    missing = missing - add_external;
                end
        
                % 如果簇外仍不足，再从簇内补足
                if missing > 0
                    local_available = numel(local_pool) - n_local;
                    add_local = min(missing, local_available);
                    n_local = n_local + add_local;
                end
        
                % -----------------------------------------------------
                % 簇内选择：优先选择本簇中适应度较好的个体
                % -----------------------------------------------------
                local_sel = [];
        
                if n_local > 0
                    [~, local_rank] = sort(fx(local_pool), 'ascend');
        
                    % 从簇内较优的候选池随机选择
                    local_top_num = max( ...
                        n_local, ...
                        ceil(0.5 * numel(local_pool)));
        
                    local_top_num = min( ...
                        local_top_num, ...
                        numel(local_pool));
        
                    local_elite_pool = ...
                        local_pool(local_rank(1:local_top_num));
        
                    rp = randperm( ...
                        numel(local_elite_pool), ...
                        n_local);
        
                    local_sel = local_elite_pool(rp);
                end
        
                % -----------------------------------------------------
                % 簇外选择：从其他搜索中心引入信息
                % -----------------------------------------------------
                external_sel = [];
        
                if n_external > 0
                    rp = randperm( ...
                        numel(external_pool), ...
                        n_external);
        
                    external_sel = external_pool(rp);
                end
        
                % 合并并随机打乱亲本顺序
                parent_sel = [
                    local_sel(:);
                    external_sel(:)
                ];
        
                if isempty(parent_sel)
                    fallback_pool = setdiff(1:NP, i);
        
                    if isempty(fallback_pool)
                        parent_sel = i;
                    else
                        parent_sel = fallback_pool( ...
                            randi(numel(fallback_pool)));
                    end
                end
        
                parent_sel = parent_sel( ...
                    randperm(numel(parent_sel)));
        
                % -----------------------------------------------------
                % 方案一：随机分块基因混洗
                % -----------------------------------------------------
                x_shuffle = pop(parent_sel(1), :);
        
                block_size = max(2, round(sqrt(D)));
                dim_order = randperm(D);
        
                for k_block = 1:block_size:D
        
                    block_dims = dim_order( ...
                        k_block:min(k_block + block_size - 1, D));
        
                    parent_id = parent_sel( ...
                        randi(numel(parent_sel)));
        
                    x_shuffle(block_dims) = ...
                        pop(parent_id, block_dims);
                end
        
                % -----------------------------------------------------
                % 局部pbest引导 + 全局差分扰动
                % -----------------------------------------------------
                v = x_shuffle ...
                    + F(i) * (Xpb(i,:) - x_shuffle) ...
                    + F(i) * (pop(r1(i),:) - popA(r2(i),:));
        
                % 二项式交叉
                [ui, mi] = lshade_cross_mask( ...
                    pop(i,:), v, CR(i), D);
        
                U(i,:) = ui;
                CrMask(i,:) = mi;
                Delta(i,mi) = v(mi) - pop(i,mi);
            end


        case 3
            % ---------------------------------------------------------
            % Action 3: mean-best引导分层变异 (HPDE风格, V2改进版)
            % 精英层: mbest引导+差分项; 非精英层: pbest策略
            % ---------------------------------------------------------
            gworstval = max(fx);
            if gworstval == gbestval
                EP = 0;
            else
                IS = abs(mean(fx) - fx) / (gworstval - gbestval);
                I = (IS > 0.1);
                EP = sum(I);
                Low_g = floor(1/10 * NP * (nfe_max - nfe) / nfe_max);
                Up_g  = ceil(1/5 * NP * (nfe_max - nfe) / nfe_max) + 1;
                if EP < Low_g || EP > Up_g
                    EP = round((Up_g - Low_g) * (nfe_max - nfe) / nfe_max + Low_g);
                end
            end
            EP = max(1, min(EP, floor(NP/2)));
            
            eIdx = ord(1:EP);
            isElite = false(NP, 1);
            isElite(eIdx) = true;
            
            sIdx = ord(EP+1:min(2*EP, NP));
            if ~isempty(sIdx)
                w_mb = NP / NP0;
                mbest = w_mb * pop(sIdx,:) + (1-w_mb) * pop(eIdx,:);
            else
                mbest = pop(eIdx,:);
            end
            
            for i = 1:NP
                if isElite(i)
                    if size(mbest,1) >= i
                        mb_i = mbest(i,:);
                    else
                        mb_i = mbest(randi(size(mbest,1)),:);
                    end
                    v = pop(i,:) + F(i)*(mb_i - pop(i,:)) ...
                        + F(i)*(pop(r1(i),:) - popA(r2(i),:));
                else
                    v = pop(i,:) + F(i)*(Xpb(i,:) - pop(i,:)) ...
                        + F(i)*(pop(r1(i),:) - popA(r2(i),:));
                end
                [ui, mi] = lshade_cross_mask(pop(i,:), v, CR(i), D);
                U(i,:) = ui;
                CrMask(i,:) = mi;
                Delta(i, mi) = v(mi) - pop(i, mi);
            end
    end
    U = lshade_repair(U, pop, Xmin, Xmax);
    
    %% --- 改进DLS：对比方向学习与方向一致性约束 ---
    if rand < tau
        for i = 1:NP
            evidence_count = LF_pos(i, :) + LF_neg(i, :);
            valid_dim = evidence_count >= dls_min_evidence;
            if cts(i) > ST && any(valid_dim)
                xi = pop(i, :);

                % 分别计算正、负平均方向；无观测维度保持为0。
                R_pos = zeros(1, D);
                R_neg = zeros(1, D);
                pos_dim = LF_pos(i, :) > 0;
                neg_dim = LF_neg(i, :) > 0;
                R_pos(pos_dim) = LV_pos(i, pos_dim) ./ LF_pos(i, pos_dim);
                R_neg(neg_dim) = LV_neg(i, neg_dim) ./ LF_neg(i, neg_dim);

                % 负方向仅在重复失败后逐渐增强，并限制其最大作用。
                lambda_i = dls_lambda_max * ...
                    min(LF_neg(i, :) / dls_neg_repeat, 1);

                % 置信度同时考虑证据数量以及正、负证据的一致程度。
                count_conf = min(evidence_count / dls_min_evidence, 1);
                contrast_num = abs(R_pos - lambda_i .* R_neg);
                contrast_den = abs(R_pos) + lambda_i .* abs(R_neg) + eps;
                C_i = count_conf .* (contrast_num ./ contrast_den);
                C_i(~valid_dim) = 0;

                R_i = C_i .* (R_pos - lambda_i .* R_neg);

                % 若学习修正与本代普通DE位移相反，则移除冲突分量。
                de_direction = U(i, :) - xi;
                R_i(R_i .* de_direction < 0) = 0;

                % 只有保留了有效一致方向时才替换普通试验。
                if ~any(abs(R_i) > 0)
                    continue;
                end

                ui_base = xi + F(i) * (Xpb(i, :) - xi) + F(i) * R_i;
                cr_lv = min(0.95, max(0.2, CR(i) + lv_cr_boost));
                [ui_lv, mi_lv] = lshade_cross_mask(xi, ui_base, cr_lv, D);
                U(i, :) = ui_lv;
                CrMask(i, :) = mi_lv;
                Delta(i, :) = 0;
                % 记录真实进入试验向量的供体位移。
                Delta(i, mi_lv) = ui_base(mi_lv) - xi(mi_lv);
            end
        end
        U = lshade_repair(U, pop, Xmin, Xmax);
    end

    % 以最终经过边界修复并实际参与评价的位移更新方向证据。
    actual_step = U - pop;
    Delta(:) = 0;
    Delta(CrMask) = actual_step(CrMask);
    
    %% ===== V3改进：真实评估（严格遵守最大FES） =====
    pop_old = pop;
    fx_old = fx;
    Y_old = Yfit;
    newf = Inf(NP, 1);
    nEval = min(NP, nfe_max - nfe);
    evalIds = (1:nEval)';
    vals = feval(fhd, U(evalIds, :)', fid);
    newf(evalIds) = vals(:);
    nfe = nfe + nEval;
    
    %% --- 改进DLS：记录有利与不利的拒绝试验 ---
    for i = evalIds'
        m = CrMask(i, :);
        if fx_old(i) < newf(i) && newf(i) < Y_old(i)
            % 虽未优于当前父代，但优于本次局部搜索的历史参考。
            LV_pos(i, m) = LV_pos(i, m) + Delta(i, m);
            LF_pos(i, m) = LF_pos(i, m) + 1;
        elseif newf(i) >= Y_old(i)
            % 完全无改进的试验作为应规避方向的负向证据。
            LV_neg(i, m) = LV_neg(i, m) + Delta(i, m);
            LF_neg(i, m) = LF_neg(i, m) + 1;
        end
    end
    
    %% --- 选择 + 停滞计数 + DLS记忆重置 ---
    better = false(NP, 1);
    better(evalIds) = newf(evalIds) < fx_old(evalIds);
    for i = evalIds'
        if better(i)
            cts(i) = 0;
            LV_pos(i, :) = 0;
            LF_pos(i, :) = 0;
            LV_neg(i, :) = 0;
            LF_neg(i, :) = 0;
            Yfit(i) = fx_old(i);
            pop(i, :) = U(i, :);
            fx(i) = newf(i);
        else
            cts(i) = cts(i) + 1;
        end
    end
    
    imp = fx_old - newf;
    imp(~better) = 0;
    if any(better)
        A = [A; pop_old(better, :)];
    end
    
    [fv, bi] = min(fx);
    if fv < gbestval
        gbestval = fv;
        gbest = pop(bi, :);
        last_imp_nfe = nfe;
    end
    
    %% --- V3改进：Q-learning奖励更新（4状态版） ---
    if gbestval < old_gbestval
        stall_gen = 0;
        rel_improve = (old_gbestval - gbestval) / abs(old_gbestval + 1e-10);
        reward = 10.0 + 20.0 * rel_improve;
    else
        stall_gen = stall_gen + 1;
        avg_improve = old_avgfx - mean(fx);
        rel_avg_improve = avg_improve / (abs(old_avgfx) + 1e-10);
        if rel_avg_improve > 0
            reward = -0.5 + 2.0 * rel_avg_improve;
        else
            reward = -1.0;
        end
    end
    
    % 下一状态
    next_diversity = (GE > ge_switch);
    next_stall = (stall_gen > 3);
    next_state = next_diversity * 2 + next_stall + 1;
    
    Q_table(curr_state, curr_action) = Q_table(curr_state, curr_action) + ...
        alpha_ql * (reward + gamma_ql * max(Q_table(next_state, :)) ...
        - Q_table(curr_state, curr_action));
    
    %% --- 记忆池更新（V3: 所有动作统一参与CR更新） ---
    S = find(better);
    if ~isempty(S)
        d = imp(S);
        s = sum(d);
        if s > 0
            w = d / s;
            SF = F(S);
            SS = ColSigmaF(S);
            if w' * SF > 0
                MF(k) = (w' * (SF .^ 2)) / (w' * SF);
            end
            
            % V3改进: 所有动作统一参与CR记忆更新（Action 2已改用binomial交叉）
            SC = CR(S);
            if max(SC) > 0 && w' * SC > 0
                MCR(k) = (w' * (SC .^ 2)) / (w' * SC);
            else
                MCR(k) = -1;
            end
            
            ms = mean(SS);
            if current_phase == 1
                MSigmaF(k) = ms * 1.1;
            else
                MSigmaF(k) = ms * 0.9;
            end
            MSigmaF(k) = min(max(MSigmaF(k), 0.01), 0.2);
            
            S_sigmaF_hist = [S_sigmaF_hist; SS(:)];
            if numel(S_sigmaF_hist) > 100
                S_sigmaF_hist = S_sigmaF_hist(end-99:end);
            end
        end
        k = mod(k, H) + 1;
    end
    
    %% --- 外部档案控制 ---
    NAmax = max(round(arc_rate * NP), 10);
    if size(A, 1) > NAmax
        rp = randperm(size(A, 1), NAmax);
        A = A(rp, :);
    end
    
    %% --- 线性种群缩减 (L-SHADE) ---
    NPnew = round((NPmin - NP0) / nfe_max * nfe + NP0);
    NPnew = max(NPmin, min(NP0, NPnew));
    if NPnew < NP
        [~, ord2] = sort(fx, 'ascend');
        keep = ord2(1:NPnew);
        pop = pop(keep, :);
        fx = fx(keep);
        LV_pos = LV_pos(keep, :);
        LF_pos = LF_pos(keep, :);
        LV_neg = LV_neg(keep, :);
        LF_neg = LF_neg(keep, :);
        cts = cts(keep);
        Yfit = Yfit(keep);
        NP = NPnew;
        if size(A, 1) > round(arc_rate * NP)
            rp = randperm(size(A, 1), max(round(arc_rate * NP), 1));
            A = A(rp, :);
        end
    end
    
    %% ===== V3核心改进：个体级自适应重启 =====
    % 原理: 每个个体独立跟踪停滞代数，停滞超过阈值的个体单独重启
    % 重启方向: 向gbest附近扰动（非纯随机），保留有价值的搜索信息
    % 扰动幅度: 随搜索进度递减（前期大幅探索，后期小幅精调）
    max_restart = max(1, round(restart_max_per_gen_frac * NP));
    stall_mask = cts >= restart_stag_lim;
    if any(stall_mask) && nfe < nfe_max
        stall_indices = find(stall_mask);
        [~, stall_ord] = sort(cts(stall_indices), 'descend');
        n_restart = min([max_restart, numel(stall_indices), nfe_max - nfe]);
        restart_idx = stall_indices(stall_ord(1:n_restart));
        restart_idx = restart_idx(:)';   % 确保是行向量，避免for循环整列赋值
        
        % 扰动幅度随进度递减，最少5%搜索空间
        perturb_scale = max(0.05, 0.20 * (1 - prog));
        search_range = Xmax - Xmin;
        
        for j = restart_idx
            % 向gbest方向注入高斯扰动
            noise = search_range .* perturb_scale .* randn(1, D);
            pop(j, :) = gbest + noise;
            pop(j, :) = min(max(pop(j,:), Xmin), Xmax);
            fx(j) = feval(fhd, pop(j,:)', fid);
            nfe = nfe + 1;
            if fx(j) < gbestval
                gbestval = fx(j);
                gbest = pop(j, :);
                last_imp_nfe = nfe;
            end
            
            % 重置该个体的所有状态
            cts(j) = 0;
            LV_pos(j, :) = 0;
            LF_pos(j, :) = 0;
            LV_neg(j, :) = 0;
            LF_neg(j, :) = 0;
            Yfit(j) = fx(j);
        end
    end
    
    %% --- 日志输出 ---
    while nfe >= print_num && print_num <= nfe_max
        fprintf(fout, '%d\t%.15f\n', print_num, gbestval - targetbest(fid));
        print_num = print_num + print_interval;
    end
end

RecordT = toc;
end

%% ==================== 辅助函数 ====================

function [z, m] = lshade_cross_mask(x, v, cr, D)
% 二项交叉 (binomial crossover)
z = x;
m = rand(1, D) < cr;
jr = randi(D);
m(jr) = true;     % 确保至少一个维度来自变异向量
z(m) = v(m);
end

function X = lshade_repair(X, pop, lb, ub)
% 越界修复: 向种群中心方向反弹
L = repmat(lb, size(X, 1), 1);
U = repmat(ub, size(X, 1), 1);
X = ((X >= L) & (X <= U)) .* X ...
    + (X < L) .* ((L + pop) .* rand(size(X)) / 2) ...
    + (X > U) .* ((U + pop) .* rand(size(X)) / 2);
X = min(max(X, L), U);
end

function y = randCauchy(mu, sigma)
% 柯西分布采样（重尾分布）
y = mu + sigma * tan(pi * (rand - 0.5));
end

function y = randWavelet(mu, sigma, prog)
% 小波变异算子（Phase1使用）
shrink = max(0.15, 1 - prog);
u = 2 * rand - 1;
w = cos(5 * pi * u) .* exp(-(u^2) / 0.32);
y = mu + sigma * shrink * w;
end
