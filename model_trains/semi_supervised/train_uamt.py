from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from model_trains.basenetwork import BaseNetWork
from utilities.losses import SoftmaxMseLoss
from models.net_factory import get_model


class UAMTTrain(BaseNetWork):
    def __init__(self, epochs, benchmark, method, log_path, eta_consistency_weight):
        super().__init__(epochs, benchmark, method, log_path)
        self.eta_consistency_weight = eta_consistency_weight
        self.ema_model = get_model('u_net', class_num=self.num_classes).to(self.device)
        self.ema_model.load_state_dict(self.model.state_dict())
        self.soft_mse_loss = SoftmaxMseLoss()

    def epoch_train(self):
        iter_per_epoch = 40
        running_ce_loss = 0.0
        running_dice_loss = 0.0
        running_train_loss = 0.0
        running_consistency_loss = 0.0
        labeled_dataloader_iterator = iter(self.train_loader)

        for i, (inputs_U, labels_U) in enumerate(self.unlabeled_loader):  # (zip(train_loader, unlabeled_train_loader)):
            if i >= iter_per_epoch:
                break
            try:
                inputs_S1, labels_S1 = next(labeled_dataloader_iterator)
            except StopIteration:
                labeled_dataloader_iterator = iter(self.train_loader)
                inputs_S1, labels_S1 = next(labeled_dataloader_iterator)

            inputs_S1, labels_S1 = inputs_S1.permute(0, 1, 3, 2), labels_S1.permute(0, 2,
                                                                                    1)  # reshaping the input samples
            inputs_U, labels_U = inputs_U.permute(0, 1, 3, 2), labels_U.permute(0, 2, 1)  # reshaping the input samples

            inputs_S1, labels_S1 = inputs_S1.to(self.device), labels_S1.to(self.device)
            inputs_U, labels_U = inputs_U.to(self.device), labels_U.to(self.device)
            # Train Model 1
            outputs_1 = self.model(inputs_S1)
            un_outputs_1 = self.model(inputs_U)

            # supervised loss
            loss_ce_1 = self.ce_loss(outputs_1, labels_S1.long())
            loss_dice_1 = self.dice_loss(labels_S1.unsqueeze(1), outputs_1)
            loss_sup = 0.5 * (loss_dice_1 + loss_ce_1)

            noise = torch.clamp(torch.randn_like(inputs_U) * 0.1, -0.2, 0.2)
            ema_inputs = inputs_U + noise

            with torch.no_grad():
                ema_output = self.ema_model(ema_inputs)

            T = 8
            _, _, w, h = inputs_U.shape
            u_batch_r = inputs_U.repeat(2, 1, 1, 1)
            stride = u_batch_r.shape[0] // 2
            preds = torch.zeros([stride * T, self.num_classes, w, h]).cuda()
            for n in range(T // 2):
                ema_inputs = u_batch_r + torch.clamp(torch.randn_like(u_batch_r) * 0.1, -0.2, 0.2)
                with torch.no_grad():
                    preds[2 * stride * n:2 * stride * (n + 1)] = self.ema_model(ema_inputs)
            preds = F.softmax(preds, dim=1)
            preds = preds.reshape(T, stride, self.num_classes, w, h)
            preds = torch.mean(preds, dim=0)
            uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)

            # consistency loss
            consistency_weight = self.get_current_consistency_weight(self.epoch, self.eta_consistency_weight)
            consistency_dist = self.soft_mse_loss(un_outputs_1, ema_output)  # (batch, 2, 112,112,80)
            threshold = (0.75 + 2.5 * consistency_weight) * np.log(2)
            mask = (uncertainty < threshold).float()
            consistency_loss = torch.sum(mask * consistency_dist) / (2 * torch.sum(mask) + 1e-16)

            # total loss
            loss = loss_sup + consistency_weight * consistency_loss
            loss.backward()

            # update model parameters
            if (i + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.update_ema_variables(self.model, self.ema_model, 0.02, self.epoch // 100)

            running_train_loss += loss.item()
            running_ce_loss += loss_ce_1.item()
            running_dice_loss += loss_dice_1.item()
            running_consistency_loss += consistency_loss.item()
            self.metric.update(outputs_1, labels_S1)

            for param_group in self.optimizer.param_groups:
                lr_ = param_group['lr']
        self.scheduler.step()

        miou = self.metric.get_mIoU()
        mdice = self.metric.get_mdice()
        epoch_loss = (running_train_loss) / (iter_per_epoch)
        epoch_ce_loss = (running_ce_loss) / (iter_per_epoch)
        epoch_dice_loss = (running_dice_loss) / (iter_per_epoch)
        epoch_consistency_loss = (running_consistency_loss) / (iter_per_epoch)
        self.logger.info('{} Epoch [{:03d}/{:03d}], total_loss : {:.4f}'.
                         format(datetime.now(), self.epoch, self.total_epochs, epoch_loss))
        self.logger.info('Train loss: {}'.format(epoch_loss))
        self.writer.add_scalar('Train/Loss', epoch_loss, self.epoch)
        self.logger.info('Train ce-loss: {}'.format(epoch_ce_loss))
        self.writer.add_scalar('Train/CE-Loss', epoch_ce_loss, self.epoch)
        self.logger.info('Train dice-loss: {}'.format(epoch_dice_loss))
        self.writer.add_scalar('Train/Dice-Loss', epoch_dice_loss, self.epoch)
        self.logger.info('Train consistency-loss: {}'.format(epoch_consistency_loss))
        self.writer.add_scalar('Train/Con-Loss', epoch_consistency_loss, self.epoch)
        self.logger.info('Train Con-Loss: {}'.format(epoch_consistency_loss))
        self.writer.add_scalar('Train/IoU', miou, self.epoch)
        self.logger.info('Train IoU: {}'.format(miou))
        self.writer.add_scalar('Train/Dice', mdice, self.epoch)
        self.logger.info('Train Dice: {}'.format(mdice))
        self.writer.add_scalar('info/lr', lr_, self.epoch)
        self.logger.info('lr: {}'.format(lr_))
        self.writer.add_scalar('info/consis_weight', consistency_weight, self.epoch)
        self.logger.info('consis_weight: {}'.format(consistency_weight))
        self.writer.add_scalar('info/threshold', threshold, self.epoch)
        return
