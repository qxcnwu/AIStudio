clear all;
d0=50;% 阈值
image=imread("C:\Users\86188\PycharmProjects\MAP\PaddleCompute\PicProcess\result\T000028.jpg");
[M,N,C]=size(image);
img_f = fft2(double(image));%傅里叶变换得到频谱
img_f=fftshift(img_f);  %移到中间
m_mid=floor(M/2);%中心点坐标
n_mid=floor(N/2);  
h = zeros(M,N);%高斯低通滤波器构造
for i = 1:M
    for j = 1:N
        d = ((i-m_mid)^2+(j-n_mid)^2);
        h(i,j) = exp(-(d)/(2*(d0^2)));      
    end
end
img_lpf = h.*img_f;
img_lpf=ifftshift(img_lpf);    %中心平移回原来状态
img_lpf=uint8(real(ifft2(img_lpf)));  %反傅里叶变换,取实数部分

subplot(1,2,1);imshow(image);title('原图');
subplot(1,2,2);imshow(img_lpf);title('高斯低通滤波d=50');

clc;
clear all;
img=imread('lena.jpg');
img_noise = imnoise(img, 'gaussian', 0, 0.01);
subplot(2,2,1),imshow(img_noise);
title('原图像');
% 将低频移动到图像的中心，这个也很重要
s=fftshift(fft2(img_noise)); 
subplot(2,2,3),imshow(log(abs(s)),[]);
title('图像傅里叶变换取对数所得频谱');
% 求解变换后的图像的中心，我们后续的处理是依据图像上的点距离中心点的距离来进行处理
[a,b] = size(img);
a0 = round(a/2);
b0 = round(b/2);
d = min(a0,b0)/12;
d = d^2;
for i=1:a
    for j=1:b
        distance = (i-a0)^2+(j-b0)^2;
        if distance<d
            h = 1;
        else
            h = 0;
        end
        low_filter(i,j) = h*s(i,j);
    end
end
subplot(2,2,4),imshow(log(abs(low_filter)),[])
title('低通滤波频谱');
new_img = uint8(real(ifft2(ifftshift(low_filter))));
subplot(2,2,2),imshow(new_img,[])
title('低通滤波后的图像');

im=imread("C:\Users\86188\PycharmProjects\MAP\PaddleCompute\PicProcess\result\channel_multi_T000009.jpg");
a=im(:,:,1);
b=im(:,:,2);
c=im(:,:,3);

im=imread("C:\Users\86188\PycharmProjects\MAP\PaddleCompute\PicProcess\result\paddle_T000009.jpg");
a1=im(:,:,1);
b1=im(:,:,2);
c1=im(:,:,3);


im=imread("D:\xiazai\GOOGLE\train_and_label\lab_train\T001804.png");
a2=im(:,:,1);
b2=im(:,:,2);
c2=im(:,:,3);

mesh(a);
hold on;
mesh(b);
hold on;
mesh(c);
hold on;

mesh(a1);
hold on;
mesh(b1);
hold on;
mesh(c1);
hold on;

mesh(a2);
hold on;
mesh(b2);
hold on;
mesh(c2);
hold on;















