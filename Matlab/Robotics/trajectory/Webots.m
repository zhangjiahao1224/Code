% Webots轨迹导入MATLAB：3D可视化+数据分析
clear; clc; close all;

% 1. 读取Webots导出的轨迹CSV（替换为你的文件路径）
traj_data = readtable('webots_uav_trajectory.csv', 'Encoding','UTF-8');

% 2. 数据清洗（过滤NaN/异常值）
valid_idx = ~isnan(traj_data.("X(m)")) & ~isnan(traj_data.("Y(m)")) & ~isnan(traj_data.("Z(m)"));
t = traj_data.("时间戳(s)")(valid_idx);
x = traj_data.("X(m)")(valid_idx);
y = traj_data.("Y(m)")(valid_idx);
z = traj_data.("Z(m)")(valid_idx);
roll = traj_data.("滚转角(rad)")(valid_idx);

% 3. 核心分析：计算飞行速度、最大高度、轨迹长度
v = gradient(sqrt(x.^2 + y.^2 + z.^2))./gradient(t); % 速度（m/s）
max_alt = max(z); % 最大高度
traj_length = sum(sqrt(diff(x).^2 + diff(y).^2 + diff(z).^2)); % 轨迹总长度

% 4. 3D可视化（MATLAB优势：专业标注+动态演示）
figure('Color','w','Position',[100,100,1000,800]);
% 绘制轨迹曲线
plot3(x, y, z, 'b-', 'LineWidth',2); hold on;
% 标记关键节点
plot3(x(1), y(1), z(1), 'ro', 'MarkerSize',8, 'DisplayName','起飞点');
plot3(x(z==max_alt), y(z==max_alt), z==max_alt, 'go', 'MarkerSize',8, 'DisplayName','最高点');
plot3(x(end), y(end), z(end), 'ko', 'MarkerSize',8, 'DisplayName','降落点');

% 美化+标注分析结果
xlabel('X 位置 (m)','FontSize',12);
ylabel('Y 位置 (m)','FontSize',12);
zlabel('Z 高度 (m)','FontSize',12);
title(sprintf('Webots轨迹分析（最大高度：%.2fm，总长度：%.2fm）',max_alt,traj_length),...
    'FontSize',16,'FontWeight','bold');
grid on; legend('Location','best');
view(45,30); camlight; lighting gouraud; axis equal;

% 5. 可选：MATLAB动态轨迹演示（比仿真平台更灵活）
figure('Color','w','Position',[1100,100,1000,800]);
h = plot3(x(1), y(1), z(1), 'r-','LineWidth',2);
h_point = plot3(x(1), y(1), z(1), 'ro','MarkerSize',6);
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('MATLAB动态轨迹演示'); grid on; axis equal;
for i = 1:length(t)
    set(h, 'XData',x(1:i), 'YData',y(1:i), 'ZData',z(1:i));
    set(h_point, 'XData',x(i), 'YData',y(i), 'ZData',z(i));
    drawnow limitrate; pause(0.01);
end