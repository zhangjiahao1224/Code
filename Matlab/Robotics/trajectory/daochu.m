% 实物飞行器真实飞行轨迹导出与可视化
clear; clc; close all;

%% 1. 读取导出的真实轨迹CSV文件（核心步骤）
% 替换为你自己的CSV文件路径（注意：路径用/或\\，不要用\）
csv_path = 'D:\DJI_Flight_Record\20260124_flight.csv';  
% 读取CSV数据，自动识别表头（大疆CSV表头包含：经度,纬度,相对高度,时间等）
flight_data = readtable(csv_path);  

% 提取核心字段（根据你的CSV表头调整字段名，以下是大疆默认字段示例）
lon = flight_data.经度;   % 经度（单位：度）
lat = flight_data.纬度;   % 纬度（单位：度）
alt = flight_data.相对高度;  % 高度（单位：米，也可用绝对高度）

% 过滤无效数据（如GPS信号丢失的NaN值）
valid_idx = ~isnan(lon) & ~isnan(lat) & ~isnan(alt);
lon = lon(valid_idx);
lat = lat(valid_idx);
alt = alt(valid_idx);

%% 2. 经纬度转平面坐标（UTM投影，关键！）
% MATLAB的deg2utm函数将经纬度转换为UTM坐标系的X（东向）、Y（北向）坐标
% 输出：x=东向距离(m), y=北向距离(m), zone=UTM带号, hem=半球(N/S)
[x, y, zone, hem] = deg2utm(lat, lon);  

% 坐标中心化（可选，让轨迹起点在原点，便于观察相对位置）
x = x - x(1);  % X轴以起点为0
y = y - y(1);  % Y轴以起点为0
alt = alt - alt(1);  % 高度以起点为基准（可选）

%% 3. 绘制真实三维飞行轨迹
figure('Color','w','Position',[100,100,1000,800]);
plot3(x, y, alt, 'b-', 'LineWidth', 2);  % 三维轨迹曲线
hold on;

% 标记关键节点：起点、最高点、终点
[max_alt, max_idx] = max(alt);
plot3(x(1), y(1), alt(1), 'ro', 'MarkerSize', 8, 'DisplayName', '起飞点');
plot3(x(max_idx), y(max_idx), alt(max_idx), 'go', 'MarkerSize', 8, 'DisplayName', '最高点');
plot3(x(end), y(end), alt(end), 'ko', 'MarkerSize', 8, 'DisplayName', '降落点');

%% 4. 图形美化（与模拟轨迹一致，增强可读性）
xlabel('东向距离 X (m)','FontSize',12);
ylabel('北向距离 Y (m)','FontSize',12);
zlabel('相对高度 Z (m)','FontSize',12);
title('实物飞行器真实三维飞行轨迹','FontSize',16,'FontWeight','bold');
grid on;
legend('Location','best','FontSize',10);
view(45, 30);  % 三维视角
camlight; lighting gouraud;  % 增强立体感
axis equal;

%% 可选：导出轨迹数据（便于后续分析）
% 将转换后的X/Y/Z数据保存为新的CSV，供其他软件使用
output_data = table(x, y, alt);
writetable(output_data, 'D:\flight_trajectory_xyz.csv');
disp('轨迹数据已导出为CSV文件！');