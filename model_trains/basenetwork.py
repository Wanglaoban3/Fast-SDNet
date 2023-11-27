import os
import random
import cv2
import numpy as np
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from models.net_factory import get_model
from utilities.metrics import TotalDiceIou
from utilities.losses import DiceLoss
from utilities.logger import get_logger
import torch
from concurrent.futures import ThreadPoolExecutor


class BaseNetWork(object):
    def __init__(self, epochs, benchmark, method, log_path, lr=0.002):
        self.method = method
        self.benchmark = benchmark
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.accumulation_steps = 2
        self._init_logger(log_path)
        self.log_path = log_path
        self.epoch = 1
        self.best_dice = 0
        self.best_model_name = ''
        self.train_loader, self.unlabeled_loader, self.val_loader, self.test_loader = None, None, None, None

        if benchmark == 'KolektorSDD':
            self.num_classes = 2
            weight = torch.tensor([0.1, 0.9], device=self.device)
        elif benchmark == 'KolektorSDD2':
            self.num_classes = 2
            weight = torch.tensor([0.2, 0.8], device=self.device)
        elif benchmark == 'carpet':
            self.num_classes = 2
            weight = torch.tensor([0.2, 0.8], device=self.device)
        elif benchmark == 'hazelnut':
            self.num_classes = 2
            weight = torch.tensor([0.2, 0.8], device=self.device)
        elif benchmark == 'MT':
            self.num_classes = 2
            weight = torch.tensor([0.1, 0.9], device=self.device)
        # DAGM数据集有10类，因此用DAGM1代表其中的第1类
        elif 'DAGM' in benchmark:
            self.num_classes = 2
            weight = torch.tensor([0.01, 0.99], device=self.device)
        elif benchmark == 'CrackForest':
            self.num_classes = 2
            weight = torch.tensor([0.2, 0.8], device=self.device)
        elif benchmark == 'CDD':
            self.num_classes = 2
            weight = torch.tensor([0.2, 0.8], device=self.device)
        else:
            raise print('please input correct benchmark!')
        self.metric = TotalDiceIou(self.num_classes)

        # model, loss and optim
        self.total_epochs = epochs
        self.model = get_model(net_type=method, class_num=self.num_classes)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=int(epochs*0.95), eta_min=1e-5)
        self.ce_loss = CrossEntropyLoss(weight=weight)
        self.cls_loss = CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.save_vis_results = SaveVisResults(log_path, self.num_classes, num_works=4)

    def _init_logger(self, log_dir):
        self.logger = get_logger(log_dir)
        print('RUNDIR: {}'.format(log_dir))
        self.writer = SummaryWriter(log_dir)

    def epoch_val(self):
        loss, ce_loss, dice_loss = 0, 0, 0
        self.model.to(self.device)
        self.metric.reset()
        self.model.eval()
        for i, pack in enumerate(self.val_loader, start=1):
            with torch.no_grad():
                images, gts = pack
                images, gts = images.permute(0, 1, 3, 2), gts.permute(0, 2, 1)  # Reshaping the input samples
                images = images.to(self.device)
                gts = gts.to(self.device)

                # model inference
                prediction = self.model(images)
                if isinstance(prediction, tuple):
                    prediction = prediction[0]

            # calculate loss
            loss_ce_1 = self.ce_loss(prediction, gts.long())
            loss_dice_1 = self.dice_loss(gts.unsqueeze(1), prediction)
            val_loss = 0.5 * (loss_dice_1 + loss_ce_1)
            loss += val_loss.item()
            ce_loss += loss_ce_1.item()
            dice_loss += loss_dice_1.item()

            # update predict results
            self.metric.update(prediction, gts)
            self.save_vis_results.log_show_predicts(self.train_loader.dataset.std, self.train_loader.dataset.mean, [i, images, gts, prediction])

        epoch_loss_val = loss / len(self.val_loader)
        epoch_ce_loss_val = ce_loss / len(self.val_loader)
        epoch_dice_loss_val = dice_loss / len(self.val_loader)

        # accumulate results
        mdice = self.metric.get_mdice()
        miou = self.metric.get_mIoU()

        self.logger.info('Val loss: {}'.format(epoch_loss_val))
        self.writer.add_scalar('Validation/loss', epoch_loss_val, self.epoch)
        self.logger.info('Val CE loss: {}'.format(epoch_ce_loss_val))
        self.writer.add_scalar('Validation/ce-loss', epoch_ce_loss_val, self.epoch)
        self.logger.info('Val Dice loss: {}'.format(epoch_dice_loss_val))
        self.writer.add_scalar('Validation/dice-loss', epoch_dice_loss_val, self.epoch)
        self.logger.info('Validation dice : {}'.format(mdice))
        self.writer.add_scalar('Validation/mDice', mdice, self.epoch)
        self.logger.info('Validation IoU : {}'.format(miou))
        self.writer.add_scalar('Validation/mIoU', miou, self.epoch)

        # save model
        if self.best_dice < mdice:
            self.best_dice = mdice
            # remove the last best model
            if self.best_model_name != '':
                os.remove(self.best_model_name)
            if not os.path.exists(self.log_path):
                os.makedirs(self.log_path)
            torch.save(self.model.state_dict(), self.log_path + f'/{self.method}_{self.benchmark}_dice_{self.best_dice}.pth')
            self.best_model_name = self.log_path + f'/{self.method}_{self.benchmark}_dice_{self.best_dice}.pth'

        self.logger.info(
            'current best dice coef: model: {}'.format(self.best_dice))

    def epoch_train(self):
        pass

    def final_test(self):
        self.model.load_state_dict(torch.load(self.best_model_name))
        for i, pack in enumerate(self.test_loader, start=1):
            with torch.no_grad():
                images, gts = pack
                images, gts = images.permute(0, 1, 3, 2), gts.permute(0, 2, 1)  # Reshaping the input samples
                images = images.to(self.device)
                gts = gts.to(self.device)

                # model inference
                prediction = self.model(images)
                if isinstance(prediction, tuple):
                    prediction = prediction[0]

            # update predict results
            self.metric.update(prediction, gts)
            self.save_vis_results.log_show_predicts(self.test_loader.dataset.std, self.test_loader.dataset.mean, [i, images, gts, prediction])

        # accumulate results
        mdice = self.metric.get_mdice()
        miou = self.metric.get_mIoU()
        self.logger.info('Test dice : {}'.format(mdice))
        self.logger.info('Test IoU : {}'.format(miou))

    def run(self, train_loader, unlabeled_loader=None, val_loader=None, test_loader=None, checkpoint=''):
        self.train_loader, self.unlabeled_loader, self.val_loader, self.test_loader = train_loader, unlabeled_loader, val_loader, test_loader
        # you only need edit epoch_train code to finish your network.
        self.model.to(self.device)
        self.logger.info(
            "train_loader {} val_loader {} test_loader {} ".format(len(self.train_loader),
                                                                                       len(self.val_loader),
                                                                                       len(self.test_loader)))
        print("Training process started!")
        print("===============================================================================================")
        for epoch in range(1, self.total_epochs):
            if self.train_loader:
                self.metric.reset()
                self.epoch_train()
                torch.cuda.empty_cache()
            if self.val_loader:
                self.metric.reset()
                self.epoch_val()
            self.epoch += 1
            print('================================================================================================')
            print('================================================================================================')
        if self.test_loader:
            self.metric.reset()
            self.final_test()

    def get_current_consistency_weight(self, epoch, eta_consistency):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return eta_consistency * self.sigmoid_rampup(epoch, int(self.total_epochs * 0.8))

    @staticmethod
    def sigmoid_rampup(current, rampup_length):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))

    @staticmethod
    def update_ema_variables(model, ema_model, alpha, global_step):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            # ema_model(t) = a*ema_model(t-1) + (1-a)*model(t)
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    @staticmethod
    def generate_mix_data(data, target, mode='cutmix', p=0.2):  # For mixing in the same loader (within labeled samples)
        batch_size, _, im_h, im_w = data.shape
        device = data.device

        new_data = []
        new_target = []
        # new_logits = []
        for i in range(batch_size):

            if mode == 'cutmix':
                mix_mask = generate_cutout_mask([im_h, im_w]).to(device)
            if random.random() < p:
                new_data.append((data[i] * mix_mask + data[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
                new_target.append((target[i] * mix_mask + target[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
                # new_logits.append((logits[i] * mix_mask + logits[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
            else:
                new_data.append((data[i].unsqueeze(0)))
                new_target.append((target[i]).unsqueeze(0))
                # new_logits.append((logits[i] * mix_mask + logits[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))

        new_data, new_target = torch.cat(new_data), torch.cat(new_target)
        return new_data, new_target.long()

    @staticmethod
    def generate_crossmix_data(data_l, data_wk, data_st, mode='cutmix', p=0.3):
        batch_size, _, im_h, im_w = data_l.shape
        device = data_l.device

        new_data_wk = []
        new_data_st = []
        for i in range(batch_size):

            if mode == 'cutmix':
                mix_mask = generate_cutout_mask([im_h, im_w]).to(device)

            if random.random() < p:
                new_data_wk.append((data_wk[i] * mix_mask + data_l[i] * (1 - mix_mask)).unsqueeze(0))
                new_data_st.append((data_st[i] * mix_mask + data_l[i] * (1 - mix_mask)).unsqueeze(0))
            else:
                new_data_wk.append((data_wk[i]).unsqueeze(0))
                new_data_st.append((data_st[i]).unsqueeze(0))
            # new_logits.append((logits[i] * mix_mask + logits[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))

        new_data_wk, new_data_st = torch.cat(new_data_wk), torch.cat(new_data_st)
        return new_data_wk, new_data_st


def generate_cutout_mask(img_size, ratio=2):  # ratio=random.choice([2,3,4,5])

    cutout_area = img_size[0] * img_size[1] / ratio

    w = np.random.randint(img_size[1] / ratio + 1, img_size[1])
    h = np.round(cutout_area / w)

    x_start = np.random.randint(0, img_size[1] - w + 1)
    y_start = np.random.randint(0, img_size[0] - h + 1)

    x_end = int(x_start + w)
    y_end = int(y_start + h)

    mask = torch.ones(img_size)
    mask[y_start:y_end, x_start:x_end] = 0
    return mask.float()


# 保存图片结果
class SaveVisResults:
    def __init__(self, log_path, num_classes, num_works):
        # 其他初始化操作
        self.executor = ThreadPoolExecutor(num_works)  # 创建线程池
        os.makedirs(os.path.join(log_path, 'test_predicts'), exist_ok=True)
        self.log_path = log_path
        self.num_classes = num_classes
        # 各类别色表
        self.color_mapping = {
            # 0: [192, 192, 192],  # Class 0 - Light gray
            1: [255, 0, 0],  # Class 1 - Red
            2: [0, 255, 0],  # Class 2 - Green
            3: [0, 0, 255],  # Class 3 - Blue
            4: [255, 255, 0],  # Class 4 - Yellow
            5: [255, 0, 255],  # Class 5 - Magenta
            6: [0, 255, 255]  # Class 6 - Cyan
        }

    def log_show_predicts(self, std, mean, batch):
        index, imgs, masks, predicts = batch
        b, c, h, w = imgs.shape
        # 所有的图片都是默认以三通道输入的，但某些单通道的数据集std和mean设置的列表长度是1，因此复制成3长度
        std = std if len(std) == 3 else std * 3
        mean = mean if len(mean) == 3 else mean * 3
        # de-normalize
        for i in range(len(std)):
            imgs[:, i] = imgs[:, i] * std[i] + mean[i]
        imgs = imgs.permute(0, 2, 3, 1).cpu().numpy() * 255  # shape: [B, C, H, W] -> [B, H, W, C]
        imgs = imgs.astype(np.uint8)
        masks = masks.cpu().numpy()  # shape: [B, H, W]
        predicts = torch.argmax(predicts, dim=1).cpu().numpy()  # shape: [B, class, H, W] -> [B, H, W]

        # Convert annotation mask to a colored image
        mask_colors = np.zeros((b, h, w, 3), dtype=np.uint8)
        predict_colors = np.zeros((b, h, w, 3), dtype=np.uint8)
        for class_id in range(1, self.num_classes):
            mask_colors[np.where(masks == class_id)] = self.color_mapping[class_id]
            predict_colors[np.where(predicts == class_id)] = self.color_mapping[class_id]
        mask_colors = mask_colors.astype(float)
        predict_colors = predict_colors.astype(float)

        # 使用多线程保存图片，传入图片，真实类别掩码，预测类别掩码
        for id, (img, mask_color, predict_color) in enumerate(zip(imgs, mask_colors, predict_colors)):
            self.executor.submit(self.save_image, id, img, mask_color, predict_color, index, b)
        return

    def save_image(self, id, img, mask_color, predict_color, index, b):
        # Scale the foreground image by the opacity (0.0 to 1.0)
        mask_color = cv2.multiply(mask_color, 0.75).astype(np.uint8)
        predict_color = cv2.multiply(predict_color, 0.75).astype(np.uint8)
        # Subtract the scaled foreground image from the background image
        merged_image = cv2.subtract(img, mask_color)
        predict_color = cv2.subtract(img, predict_color)
        # Concatenate resized annotation color image and prediction mask horizontally
        merged_image = np.concatenate((merged_image, predict_color), axis=1)
        merged_image = cv2.cvtColor(merged_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.log_path, 'test_predicts', f'{int(index * b + id)}.jpg'), merged_image)

    def cleanup(self):
        self.executor.shutdown()  # 关闭线程池，确保所有任务都完成
