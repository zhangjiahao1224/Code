%% VLA视觉避障轨迹优化（彻底修复运算符错误，适配低版本MATLAB）
clear; clc; close all;

% ===================== 0. 初始化RTB+OMPL（解决路径遮蔽+接口加载） =====================
% 替换为你的RTB根路径
rtb_root = 'C:\Users\17734\AppData\Roaming\MathWorks\MATLAB Add-Ons\Toolboxes\Robotics Toolbox for MATLAB';
rmpath(genpath(rtb_root));
addpath('top', genpath(rtb_root));
savepath;
startup_rvc; % 关键：加载RTB+OMPL接口（ob/og类）

% ===================== 1. 加载基础轨迹和关键变量 =====================
% 加载MATLAB生成的基础轨迹（需先运行基础轨迹代码生成该文件）
if ~exist('uav_base_traj.csv', 'file')
    error('❌ 未找到基础轨迹文件！请先运行无人机基础轨迹生成代码');
end
traj_data = readmatrix('uav_base_traj.csv');
t = traj_data(:,1);          % 时间戳
traj_xyz = traj_data(:,2:4); % 3D位置轨迹
traj_yaw = traj_data(:,5);   % 航向角（补全原代码缺失的变量）
t_total = max(t);            % 总飞行时间（补全原代码缺失的变量）

% 重新定义航点（与基础轨迹一致，补全绘图用变量）
waypoints = [
    0, 0, 2, 0;     % 起点
    5, 0, 2, 0;     % 航点1
    5, 5, 3, pi/4;  % 航点2
    0, 5, 2, pi/2;  % 航点3
    0, 0, 2, 0      % 终点
];

% ===================== 2. 定义视觉检测的障碍物 =====================
% 圆柱障碍物：中心(5,2.5,2.5)，半径1m，高度4m（模拟视觉相机检测结果）
obs_center = [5, 2.5, 2.5];
obs_radius = 1;
obs_height = 4;
safe_margin = 0.5; % 安全余量（避障距离=半径+余量）

% ===================== 3. RRT*避障优化（RTB+OMPL） =====================
% 3.1 定义3D空间边界（0~10m X/Y，0~5m Z）
space = ob.RealVectorStateSpace(3);
bounds = ob.RealVectorBounds(3);
bounds.setLow([0, 0, 0.5]); % Z轴最低0.5m（离地）
bounds.setHigh([10, 10, 5]);
space.setBounds(bounds);

% 3.2 定义障碍物约束函数（解决低版本嵌套函数作用域问题）
% 封装为独立函数句柄，传递障碍物参数
isStateValid = @(state) checkObstacle(state, obs_center, obs_radius, safe_margin);

% 3.3 初始化规划器
ss = og.SimpleSetup(space);
ss.setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid));

% 3.4 设置起点/终点（与基础轨迹一致）
start = ob.State(space); start.setValues(traj_xyz(1,:));
goal = ob.State(space); goal.setValues(traj_xyz(end,:));
ss.setStartAndGoalStates(start, goal);

% 3.5 配置RRT*规划器（低版本兼容）
planner = og.RRTstar(ss.getSpaceInformation());
planner.setRange(0.5); % 采样步长（越小路径越平滑）
ss.setPlanner(planner);
ss.setup();

% 3.6 求解避障路径（带容错处理，修复运算符错误）
disp('🔍 正在运行RRT*避障规划...');
solve_status = ss.solve(5.0); % 求解5秒
% 修正：MATLAB用~=表示不等于，且低版本用数字1替代ob.SOLVED（避免枚举错误）
if solve_status ~= 1  % ob.SOLVED的枚举值对应数字1，低版本直接用数字更稳定
    warning('⚠️ RRT*求解失败，使用基础轨迹替代！');
    opt_traj_xyz = traj_xyz; % 求解失败时用基础轨迹
else
    % 提取优化后路径并采样
    opt_traj = ss.getSolutionPath();
    opt_traj.interpolate(200); % 插值为200个点（与基础轨迹采样数一致）
    opt_xyz = opt_traj.getStates();
    % 修正：低版本MATLAB cellfun需指定输出类型
    opt_xyz_cell = cellfun(@(s) s.getValues(), opt_xyz, 'UniformOutput', false);
    opt_xyz = cell2mat(opt_xyz_cell);
    
    % 3.7 多项式平滑（保形插值，保证轨迹平滑）
    opt_t = linspace(0, t_total, size(opt_xyz,1));
    opt_traj_xyz = zeros(length(t), 3);
    for i = 1:3
        opt_traj_xyz(:,i) = interp1(opt_t, opt_xyz(:,i), t, 'pchip');
    end
    disp('✅ RRT*避障路径求解成功！');
end

% ===================== 4. 可视化对比（基础轨迹vs优化轨迹） =====================
figure('Name','VLA视觉避障优化轨迹');
% 绘制基础轨迹（灰色虚线）
plot3(traj_xyz(:,1), traj_xyz(:,2), traj_xyz(:,3), 'g--', 'LineWidth', 1, 'DisplayName','基础轨迹');
hold on;
% 绘制优化后轨迹（红色实线）
plot3(opt_traj_xyz(:,1), opt_traj_xyz(:,2), opt_traj_xyz(:,3), 'r-', 'LineWidth', 2, 'DisplayName','VLA优化轨迹');
% 绘制障碍物（半透明圆柱）
[X,Y,Z] = cylinder(obs_radius, 50);
X = X + obs_center(1); Y = Y + obs_center(2); Z = Z * obs_height + (obs_center(3)-obs_height/2);
surf(X,Y,Z, 'FaceAlpha',0.3, 'EdgeColor','k', 'DisplayName','障碍物');
% 绘制航点
scatter3(waypoints(:,1), waypoints(:,2), waypoints(:,3), 50, 'b', 'filled', 'DisplayName','航点');
% 标注
xlabel('X(m)'); ylabel('Y(m)'); zlabel('Z(m)');
title('无人机VLA视觉避障优化轨迹'); grid on; view(3); legend;

% ===================== 5. 保存优化后轨迹（ROS兼容） =====================
opt_traj_data = [t, opt_traj_xyz, traj_yaw]; % 时间, X, Y, Z, 航向角
writematrix(opt_traj_data, 'uav_vla_traj.csv', 'TextType', 'string');
disp('✅ VLA避障轨迹已保存为uav_vla_traj.csv');

% ===================== 辅助函数：障碍物检测（独立函数，低版本兼容） =====================
function valid = checkObstacle(state, obs_center, obs_radius, safe_margin)
    pos = state.getValues(); % 获取采样点位置
    % 1. 检查与圆柱障碍物的距离
    dist = norm(pos(1:2) - obs_center(1:2)); % 水平距离
    height_ok = (pos(3) > obs_center(3)-obs_height/2) && (pos(3) < obs_center(3)+obs_height/2);
    obs_collision = (dist < obs_radius + safe_margin) && height_ok;
    % 2. 检查离地高度
    ground_collision = (pos(3) < 0.5);
    % 3. 有效状态：无碰撞
    valid = ~obs_collision && ~ground_collision;
end