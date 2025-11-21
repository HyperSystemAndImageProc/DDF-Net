import matplotlib.pyplot as plt

model = load() # 读训练好的模型

t  =               #生成时间戳


xt =  #读序列


#1 训练集合内图
_, recon_from_xt = diffusion.reconstruct(model, xt=xt, tempT=t, num=5)  # 重建数据
#

2 集合外
pure_noise =  torch.randn(shape, device=device)  # 从随机噪声开始
_, recon_from_xt = diffusion.reconstruct(model, xt=pure_noise, tempT=t, num=5)  # 重建数据


plt.draw(recon_from_xt)



# 3 老店

# 假设真实脑电噪声 brain_noise  <======> real_brain
bb  = brain_noise + 0.001*torch.randn
_, recon_from_xt = diffusion.reconstruct(model, xt=bb, tempT=t, num=5, from_noise=True)  # 重建数据

plot(recon_from_xt)
plot(real_brain)



diffusion.reconstruct(model, xt=brain_noise, tempT=t, num=5, from_noise=True)  # 重建数据
