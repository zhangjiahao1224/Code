% 基于MATLAB的三维飞行轨迹绘制（修复动态绘图对象失效错误）
clear; clc; close all;  % 清空工作区、命令行，关闭已有图形窗口

%% 1. 生成飞行轨迹数据
t = linspace(0, 60, 1000);  % 时间序列：0到60秒，共1000个采样点
x = 5*t + 2*sin(t);  % X轴：主方向前进+小幅左右偏移
y = 3*t + 5*cos(2*(t-20)).*(t>=20);  % Y轴：20秒后开始转弯
z = zeros(size(t));  % 初始化高度数组

% 用索引定位关键时间节点（修复浮点数比较问题）
idx_10 = find(t >= 10, 1, 'first');  
idx_40 = find(t >= 40, 1, 'first');  

% 爬升阶段（0-10秒）：抛物线加速爬升
z(1:idx_10) = 0.5*t(1:idx_10).^2;        
% 平飞阶段（10-40秒）：高度保持
z(idx_10+1:idx_40) = z(idx_10);         
% 下降阶段（40-60秒）：抛物线下降
z(idx_40+1:end) = z(idx_40) - 0.3*(t(idx_40+1:end)-40).^2;  
z(z<0) = 0;  % 确保高度不为负

%% 2. 绘制静态三维飞行轨迹
figure('Color','w','Position',[100,100,1000,800]);  
plot3(x, y, z, 'b-', 'LineWidth', 2);  % 绘制三维轨迹
hold on;  

% 标记关键节点
plot3(x(1), y(1), z(1), 'ro', 'MarkerSize', 8, 'DisplayName', '起点');
plot3(x(idx_10), y(idx_10), z(idx_10), 'go', 'MarkerSize', 8, 'DisplayName', '平飞起点');
plot3(x(idx_40), y(idx_40), z(idx_40), 'mo', 'MarkerSize', 8, 'DisplayName', '下降起点');
plot3(x(end), y(end), z(end), 'ko', 'MarkerSize', 8, 'DisplayName', '终点');

% 图形美化
xlabel('X 位置 (m)','FontSize',12);  
ylabel('Y 位置 (m)','FontSize',12);  
zlabel('Z 高度 (m)','FontSize',12);  
title('三维飞行轨迹可视化','FontSize',16,'FontWeight','bold');
grid on;  
legend('Location','best','FontSize',10);  
view(45, 30);  
camlight;  
lighting gouraud;  
axis equal;  

%% 3. 修复后的动态绘制飞行轨迹（核心优化）
% 新建独立的动态绘图窗口，避免与静态窗口冲突
fig_dyn = figure('Color','w','Position',[1100,100,1000,800]);
% 先设置坐标轴范围（基于完整轨迹数据，避免初始坐标失效）
xlim([min(x), max(x)]);
ylim([min(y), max(y)]);
zlim([0, max(z)]);
grid on;
xlabel('X 位置 (m)','FontSize',12);
ylabel('Y 位置 (m)','FontSize',12);
zlabel('Z 高度 (m)','FontSize',12);
title('三维飞行轨迹动态演示','FontSize',16,'FontWeight','bold');
view(45, 30);
axis equal;

% 初始化轨迹线和飞行器位置点（用真实起点坐标，避免0,0,0失效）
h = plot3(x(1), y(1), z(1), 'r-', 'LineWidth', 2);  % 轨迹线初始化为起点
h_point = plot3(x(1), y(1), z(1), 'ro', 'MarkerSize', 6);  % 飞行器位置点
hold on;  % 保持窗口，避免对象被清空

% 循环绘制动态轨迹（添加对象有效性检查）
for i = 1:length(t)
    % 检查对象是否有效（核心修复：避免访问已删除的对象）
    if ~isvalid(h) || ~isvalid(h_point) || ~isvalid(fig_dyn)
        disp('图形窗口或对象已关闭，终止动态绘制');
        break;
    end
    
    % 更新轨迹线（逐步扩展）
    set(h, 'XData', x(1:i), 'YData', y(1:i), 'ZData', z(1:i));
    % 更新飞行器位置点（当前位置）
    set(h_point, 'XData', x(i), 'YData', y(i), 'ZData', z(i));
    
    drawnow limitrate;  % 优化刷新机制，避免卡顿
    pause(0.01);  % 控制动画速度（可调整，如0.02更慢）
end

disp('动态轨迹绘制完成！');