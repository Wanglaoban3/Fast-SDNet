import random
from datetime import datetime
import torch
from model_trains.basenetwork import BaseNetWork


class DCTTrain(BaseNetWork):
    def __init__(self, epochs, benchmark, method, log_path, eta_consistency_weight):
        super().__init__(epochs, benchmark, method, log_path)
        self.eta_consistency_weight = eta_consistency_weight

    def epoch_train(self):
        iter_per_epoch = 40
        running_ce_loss = 0.0
        running_dice_loss = 0.0
        running_train_loss = 0.0
        running_consistency_loss = 0.0

        self.optimizer.zero_grad()
        self.model.train()
        dataloader_iterator1 = iter(self.train_loader)
        for i, (inputs_U, labels_U) in enumerate(self.unlabeled_loader):  # (zip(train_loader, unlabeled_train_loader)):
            if i >= iter_per_epoch:
                break
            try:
                inputs_S1, labels_S1 = next(dataloader_iterator1)
            except StopIteration:
                dataloader_iterator1 = iter(self.train_loader)
                inputs_S1, labels_S1 = next(dataloader_iterator1)

            inputs_S1, labels_S1 = inputs_S1.permute(0, 1, 3, 2), labels_S1.permute(0, 2, 1)  # reshaping the input samples
            inputs_U, labels_U = inputs_U.permute(0, 1, 3, 2), labels_U.permute(0, 2, 1)  # reshaping the input samples
            inputs_S1, labels_S1 = inputs_S1.to(self.device), labels_S1.to(self.device)
            inputs_U, labels_U = inputs_U.to(self.device), labels_U.to(self.device)

            # Labeled samples output
            outputs = self.model(inputs_S1)

            # Unlabeled samples output
            un_outputs = self.model(inputs_U)  # Normal unlabeled outputs
            un_outputs_soft = torch.softmax(un_outputs, dim=1)

            rot_times = random.randrange(0, 4)
            rotated_unlabeled_samples = torch.rot90(inputs_U, rot_times, [2, 3])
            unlabeled_rot_outputs = self.model(rotated_unlabeled_samples)  # perturbed unlabeled ouputs
            unlabeled_rot_outputs_soft = torch.softmax(unlabeled_rot_outputs, dim=1)

            # CE_loss
            loss_ce = self.ce_loss(outputs, labels_S1.long())
            # Dice_loss
            loss_dice = self.dice_loss(labels_S1.unsqueeze(1), outputs)
            # total supervised loss
            supervised_loss = 0.5 * (loss_ce + loss_dice)

            # unlabeled loss
            consistency_weight = self.get_current_consistency_weight(self.epoch, self.eta_consistency_weight)  # Consistency weight multipliers
            consistency_loss = torch.mean(
                (unlabeled_rot_outputs_soft - torch.rot90(un_outputs_soft, rot_times, [2, 3])) ** 2)

            # total loss
            loss = supervised_loss + consistency_weight * consistency_loss
            loss.backward()

            if (i + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            running_train_loss += loss.item()
            running_ce_loss += loss_ce.item()
            running_dice_loss += loss_dice.item()
            running_consistency_loss += consistency_loss.item()
            self.metric.update(outputs, labels_S1)

            for param_group in self.optimizer.param_groups:
                lr_ = param_group['lr']
        self.scheduler.step()

        epoch_loss = (running_train_loss) / (iter_per_epoch)
        epoch_ce_loss = (running_ce_loss) / (iter_per_epoch)
        epoch_dice_loss = (running_dice_loss) / (iter_per_epoch)
        epoch_consistency_loss = (running_consistency_loss) / (iter_per_epoch)
        epoch_iou = self.metric.get_mIoU()
        epoch_dice = self.metric.get_mdice()
        self.logger.info('{} Epoch [{:03d}/{:03d}], total_loss : {:.4f}'.
                         format(datetime.now(), self.epoch, self.total_epochs, epoch_loss))

        self.logger.info('Train loss: {}'.format(epoch_loss))
        self.writer.add_scalar('Train/Loss', epoch_loss, self.epoch)

        self.logger.info('Train ce-loss: {}'.format(epoch_ce_loss))
        self.writer.add_scalar('Train/CE-Loss', epoch_ce_loss, self.epoch)

        self.logger.info('Train dice-loss: {}'.format(epoch_dice_loss))
        self.writer.add_scalar('Train/Dice-Loss', epoch_dice_loss, self.epoch)
        self.logger.info('Train consistency-loss: {}'.format(epoch_consistency_loss))
        self.writer.add_scalar('Train/Consistency-Loss', epoch_consistency_loss, self.epoch)
        self.logger.info('Train IoU: {}'.format(epoch_iou))
        self.writer.add_scalar('Train/IoU', epoch_iou, self.epoch)
        self.logger.info('Train Dice: {}'.format(epoch_dice))
        self.writer.add_scalar('Train/Dice', epoch_dice, self.epoch)
        self.logger.info('Lr: {}'.format(lr_))
        self.writer.add_scalar('info/lr', lr_, self.epoch)
        self.logger.info('consis_weight: {}'.format(consistency_weight))
        self.writer.add_scalar('info/consis_weight', consistency_weight, self.epoch)
        return
