% PX4轨迹导入MATLAB：NED转ENU+飞控数据分析
clear; clc; close all;

% 1. 读取PX4 ULog转换的CSV（局部坐标系X/Y/Z）
px4_data = readtable('vehicle_local_position_0.csv');
t = px4_data.timestamp/1e6; % 时间戳转秒（PX4默认微秒）
x_ned = px4_data.x; % NED坐标系X（北）
y_ned = px4_data.y; % NED坐标系Y（东）
z_ned = px4_data.z; % NED坐标系Z（下）

% 2. NED→ENU坐标转换（适配MATLAB可视化习惯）
x_enu = y_ned; % ENU东
y_enu = x_ned; % ENU北
z_enu = -z_ned; % ENU上（取反）

% 3. 3D可视化（突出飞控轨迹精度）
figure('Color','w');
plot3(x_enu, y_enu, z_enu, 'g-', 'LineWidth',2); hold on;
plot3(x_enu(1), y_enu(1), z_enu(1), 'ro', 'MarkerSize',8);
xlabel('ENU 东向 (m)'); ylabel('ENU 北向 (m)'); zlabel('ENU 高度 (m)');
title('PX4 SITL轨迹（NED→ENU转换）');
grid on; view(45,30); axis equal;

% 4. 飞控分析：计算轨迹跟踪误差（若有期望轨迹）
% 示例：对比MATLAB生成的期望轨迹
t_ref = linspace(0, max(t), length(t));
x_ref = 5*t_ref; y_ref = 3*t_ref; z_ref = 0.5*t_ref.^2;
error = sqrt((x_enu-x_ref).^2 + (y_enu-y_ref).^2 + (z_enu-z_ref).^2);
figure; plot(t, error, 'r-');
xlabel('时间 (s)'); ylabel('跟踪误差 (m)'); title('PX4轨迹跟踪误差');