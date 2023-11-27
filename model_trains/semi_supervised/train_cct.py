from datetime import datetime
from model_trains.basenetwork import BaseNetWork
import torch


class CCTTrain(BaseNetWork):
    def __init__(self, epochs, benchmark, method, log_path, eta_consistency_weight):
        super().__init__(epochs, benchmark, method, log_path)
        self.eta_consistency_weight = eta_consistency_weight

    def epoch_train(self):
        iter_per_epoch = 40
        running_train_ce_loss = 0.0
        running_train_dice_loss = 0.0
        running_train_loss = 0.0
        running_aux1_loss = 0.0
        running_aux2_loss = 0.0
        running_aux3_loss = 0.0

        self.optimizer.zero_grad()
        self.model.train()
        # 因为标注数据量少于未标注的数据，用这种办法可以实现标注数据集遍历完之后又重头开始遍历
        labeled_dataloader_iterator = iter(self.train_loader)
        for i, (inputs_U, labels_U) in enumerate(self.unlabeled_loader):  # (zip(train_loader, unlabeled_train_loader)):
            if i >= iter_per_epoch:
                break
            try:
                inputs_S1, labels_S1 = next(labeled_dataloader_iterator)
            except StopIteration:
                labeled_dataloader_iterator = iter(self.train_loader)
                inputs_S1, labels_S1 = next(labeled_dataloader_iterator)

            inputs_S1, labels_S1 = inputs_S1.permute(0, 1, 3, 2), labels_S1.permute(0, 2, 1)  # reshaping the input samples
            inputs_U, labels_U = inputs_U.permute(0, 1, 3, 2), labels_U.permute(0, 2, 1)  # reshaping the input samples

            inputs_S1, labels_S1 = inputs_S1.to(self.device), labels_S1.to(self.device)
            inputs_U, labels_U = inputs_U.to(self.device), labels_U.to(self.device)

            # Labeled samples output
            outputs, outputs_aux1, outputs_aux2, outputs_aux3 = self.model(inputs_S1)

            # Unlabeled samples output
            un_outputs, un_outputs_aux1, un_outputs_aux2, un_outputs_aux3 = self.model(inputs_U)
            un_outputs_soft = torch.softmax(un_outputs, dim=1)
            un_outputs_aux1_soft = torch.softmax(un_outputs_aux1, dim=1)
            un_outputs_aux2_soft = torch.softmax(un_outputs_aux2, dim=1)
            un_outputs_aux3_soft = torch.softmax(un_outputs_aux3, dim=1)

            # CE_loss
            loss_ce = self.ce_loss(outputs, labels_S1.long())
            loss_ce_aux1 = self.ce_loss(outputs_aux1, labels_S1.long())
            loss_ce_aux2 = self.ce_loss(outputs_aux2, labels_S1.long())
            loss_ce_aux3 = self.ce_loss(outputs_aux3, labels_S1.long())

            # Dice_loss
            loss_dice = self.dice_loss(labels_S1.unsqueeze(1), outputs)
            loss_dice_aux1 = self.dice_loss(labels_S1.unsqueeze(1), outputs_aux1)
            loss_dice_aux2 = self.dice_loss(labels_S1.unsqueeze(1), outputs_aux2)
            loss_dice_aux3 = self.dice_loss(labels_S1.unsqueeze(1), outputs_aux3)

            # ce_loss and dice_loss
            loss_main = 0.5 * (loss_ce + loss_dice)
            loss_aux1 = 0.5 * (loss_ce_aux1 + loss_dice_aux1)
            loss_aux2 = 0.5 * (loss_ce_aux2 + loss_dice_aux2)
            loss_aux3 = 0.5 * (loss_ce_aux3 + loss_dice_aux3)

            # Total supervised losss
            total_loss_ce = (loss_ce + loss_ce_aux1 + loss_ce_aux2 + loss_ce_aux3) / 4  # for plotting epoch loss
            total_loss_dice = (loss_dice + loss_dice_aux1 + loss_dice_aux2 + loss_dice_aux3) / 4  # for plotting epoch loss
            supervised_loss = (loss_main + loss_aux1 + loss_aux2 + loss_aux3) / 4  # for plotting epoch loss

            # unsupervised loss --> consistency
            consistency_loss_aux1 = torch.mean((un_outputs_soft - un_outputs_aux1_soft) ** 2)
            consistency_loss_aux2 = torch.mean((un_outputs_soft - un_outputs_aux2_soft) ** 2)
            consistency_loss_aux3 = torch.mean((un_outputs_soft - un_outputs_aux3_soft) ** 2)

            consistency_loss = (consistency_loss_aux1 + consistency_loss_aux2 + consistency_loss_aux3) / 3
            consistency_weight = self.get_current_consistency_weight(self.epoch, self.eta_consistency_weight)
            loss = supervised_loss + consistency_weight * consistency_loss
            loss.backward()

            if (i + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            running_train_loss += loss.item()
            running_train_ce_loss += total_loss_ce.item()
            running_train_dice_loss += total_loss_dice.item()
            running_aux1_loss += loss_ce_aux1.item()
            running_aux2_loss += loss_ce_aux2.item()
            running_aux3_loss += loss_ce_aux3.item()

            self.metric.update(outputs, labels_S1)

            for param_group in self.optimizer.param_groups:
                lr_ = param_group['lr']
        self.scheduler.step()

        epoch_loss = (running_train_loss) / (iter_per_epoch)
        epoch_ce_loss = (running_train_ce_loss) / (iter_per_epoch)
        epoch_dice_loss = (running_train_dice_loss) / (iter_per_epoch)
        epoch_aux1_loss = (running_aux1_loss) / (iter_per_epoch)
        epoch_aux2_loss = (running_aux2_loss) / (iter_per_epoch)
        epoch_aux3_loss = (running_aux3_loss) / (iter_per_epoch)

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
        self.logger.info('Train aux1-loss: {}'.format(epoch_aux1_loss))
        self.writer.add_scalar('Train/aux1', epoch_aux1_loss, self.epoch)
        self.logger.info('Train aux2-loss: {}'.format(epoch_aux2_loss))
        self.writer.add_scalar('Train/aux2', epoch_aux2_loss, self.epoch)
        self.logger.info('Train aux3-loss: {}'.format(epoch_aux3_loss))
        self.writer.add_scalar('Train/aux3', epoch_aux3_loss, self.epoch)
        self.logger.info('Train IoU: {}'.format(epoch_iou))
        self.writer.add_scalar('Train/IoU', epoch_iou, self.epoch)
        self.logger.info('Train Dice: {}'.format(epoch_dice))
        self.writer.add_scalar('Train/Dice', epoch_dice, self.epoch)
        self.logger.info('Lr: {}'.format(lr_))
        self.writer.add_scalar('info/lr', lr_, self.epoch)
        self.logger.info('consis_weight: {}'.format(consistency_weight))
        self.writer.add_scalar('info/consis_weight', consistency_weight, self.epoch)
        return
