% AirSim/OpenUAV VLN轨迹导入MATLAB：指令-轨迹关联分析
clear; clc; close all;

% 1. 读取VLN轨迹CSV（含语言指令ID）
vln_data = readtable('airsim_vln_trajectory.csv', 'Encoding','UTF-8');
t = vln_data.("时间戳(s)");
x = vln_data.("X(m)"); y = vln_data.("Y(m)"); z = vln_data.("Z(m)");
instr_id = vln_data.("VLN指令ID");

% 2. 按指令分段可视化轨迹
figure('Color','w'); hold on;
% 提取不同指令对应的轨迹段
instr_001_idx = strcmp(instr_id, 'INS_001'); % "飞向红顶建筑"
instr_002_idx = strcmp(instr_id, 'INS_002'); % "绕建筑盘旋"
plot3(x(instr_001_idx), y(instr_001_idx), z(instr_001_idx), 'b-', 'LineWidth',2, 'DisplayName','INS_001');
plot3(x(instr_002_idx), y(instr_002_idx), z(instr_002_idx), 'r-', 'LineWidth',2, 'DisplayName','INS_002');

xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('VLN指令-轨迹关联可视化');
grid on; legend('Location','best'); view(45,30);

% 3. 量化分析：指令执行时长、轨迹效率
instr_001_time = max(t(instr_001_idx)) - min(t(instr_001_idx));
instr_001_length = sum(sqrt(diff(x(instr_001_idx)).^2 + diff(y(instr_001_idx)).^2));
disp(['指令INS_001执行时长：',num2str(instr_001_time),'s，轨迹长度：',num2str(instr_001_length),'m']);