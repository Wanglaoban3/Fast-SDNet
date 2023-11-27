import os
from datetime import datetime
import torch
from model_trains.basenetwork import BaseNetWork
from utilities.metrics import F1PR


# 专门用来训练分类模型
class ClsModelTrain(BaseNetWork):
    def __init__(self, epochs, benchmark, method, log_path, lr):
        super().__init__(epochs, benchmark, method, log_path, lr=lr)
        self.metric = F1PR()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=int(epochs*0.95), eta_min=1e-5)
        self.best_f1 = 0
        self.defect_bank = []
        self.ce_loss.weight = None

    def epoch_train(self):
        self.optimizer.zero_grad()
        self.model.train()
        running_loss = 0
        for i, (images, gts) in enumerate(self.train_loader):  # (zip(train_loader, unlabeled_train_loader)):
            images, gts = images.permute(0, 1, 3, 2), gts.permute(0, 2, 1)  # reshaping the input samples
            images, gts = images.to(self.device), gts.to(self.device)
            cls_output = self.model(images)
            # 获取每个输入对应的label，因为每张mask都只包含一个除背景以外的类别
            # 分类只用做二分类即可
            b, _, _ = gts.shape
            cls_label = gts.reshape(b, -1).sum(dim=1)
            cls_label[cls_label>0] = 1
            loss = self.cls_loss(cls_output, cls_label.long())
            loss.backward()
            if (i + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.metric.update(cls_output, cls_label)
            running_loss += loss.item()

        running_loss = running_loss / len(self.train_loader)
        results = self.metric.get_f1_pr(mode='fixed')
        self.logger.info('{} Epoch [{:03d}/{:03d}], total_loss : {:.4f}'.
                         format(datetime.now(), self.epoch, self.total_epochs, running_loss))
        self.writer.add_scalar('Train/Loss', running_loss, self.epoch)
        self.logger.info('Train_F1: {}'.format(results['f1']))
        self.writer.add_scalar('Train/F1', results['f1'], self.epoch)
        self.logger.info('Train_Precision: {}'.format(results['best_p']))
        self.writer.add_scalar('Train/Precision', results['best_p'], self.epoch)
        self.logger.info('Train Recall: {}'.format(results['best_r']))
        self.writer.add_scalar('Train/Recall', results['best_r'], self.epoch)

    def epoch_val(self):
        self.model.eval()
        val_loss = 0.0
        for i, pack in enumerate(self.val_loader, start=1):
            with torch.no_grad():
                images, gts = pack
                images, gts = images.permute(0, 1, 3, 2), gts.permute(0, 2, 1)  # Reshaping the input samples
                images = images.to(self.device)
                gts = gts.to(self.device)
                cls_output = self.model(images)
                # 获取每个输入对应的label，因为每张mask都只包含一个除背景以外的类别
                # 分类只用做二分类即可
                b, _, _ = gts.shape
                cls_label = gts.reshape(b, -1).sum(dim=1)
                cls_label[cls_label > 0] = 1
                cls_loss = self.ce_loss(cls_output, cls_label.long())
                self.metric.update(cls_output, cls_label)
                val_loss += cls_loss.item()

        epoch_cls_loss = val_loss / len(self.val_loader)
        results = self.metric.get_f1_pr(mode='fixed')
        self.logger.info('Val cls loss: {}'.format(epoch_cls_loss))
        self.writer.add_scalar('Val/Cls_Loss', epoch_cls_loss, self.epoch)
        self.logger.info('Val F1: {}'.format(results['f1']))
        self.writer.add_scalar('Val/F1', results['f1'])
        self.logger.info('Val Precision: {}'.format(results['best_p']))
        self.writer.add_scalar('Val/Precision', results['best_p'])
        self.logger.info('Val Recall: {}'.format(results['best_r']))
        self.writer.add_scalar('Val/Recall', results['best_r'])

        # save model
        if self.best_f1 < results['f1']:
            self.best_f1 = results['f1']
            # remove the last best model
            if self.best_model_name != '':
                os.remove(self.best_model_name)
            if not os.path.exists(self.log_path):
                os.makedirs(self.log_path)
            torch.save(self.model.state_dict(), self.log_path + f'/{self.method}_{self.benchmark}_dice_{self.best_dice}_f1_{self.best_f1}.pth')
            self.best_model_name = self.log_path + f'/{self.method}_{self.benchmark}_dice_{self.best_dice}_f1_{self.best_f1}.pth'

        self.logger.info(
            'current best f1 coef: model: {}'.format(self.best_f1))

    def final_test(self):
        self.model.load_state_dict(torch.load(self.best_model_name))
        for i, pack in enumerate(self.test_loader, start=1):
            with torch.no_grad():
                images, gts = pack
                images, gts = images.permute(0, 1, 3, 2), gts.permute(0, 2, 1)  # Reshaping the input samples
                images = images.to(self.device)
                gts = gts.to(self.device)

                # model inference
                cls_output = self.model(images)
                b, _, _ = gts.shape
                cls_label = gts.reshape(b, -1).sum(dim=1)
                cls_label[cls_label > 0] = 1
                self.metric.update(cls_output, cls_label)

        # accumulate results
        results = self.metric.get_f1_pr(mode='fixed')
        self.metric.draw(self.log_path + f'/{self.method}_{self.benchmark}_{self.best_dice}_Test.png')
        self.logger.info('Test F1: {}'.format(results['f1']))
        self.writer.add_scalar('Test/F1', results['f1'])
        self.logger.info('Test Precision: {}'.format(results['best_p']))
        self.writer.add_scalar('Test/Precision', results['best_p'])
        self.logger.info('Test Recall: {}'.format(results['best_r']))
        self.writer.add_scalar('Test/Recall', results['best_r'])
