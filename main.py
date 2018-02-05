"""
 Between-class Learning for Image Classification.
 Yuji Tokozume, Yoshitaka Ushiku, and Tatsuya Harada
"""
import os

import matplotlib
matplotlib.use('Agg')  # NOQA

import chainer
from chainer import optimizers
from chainer import training
from chainer.training import extensions
from chainer.training.triggers import MaxValueTrigger, ManualScheduleTrigger, MinValueTrigger

import dataset
import net
import opts


def main():
    opt = opts.parse()
    model = net.ConvNet(opt.n_classes, opt.BC, opt.nobias, opt.dropout_ratio)
    if opt.gpu > -1:
        chainer.cuda.get_device_from_id(opt.gpu).use()
        model.to_gpu()
    optimizer = optimizers.NesterovAG(lr=opt.LR, momentum=opt.momentum)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(opt.weight_decay))
    train_iter, val_iter = dataset.setup(opt)
    updater = training.StandardUpdater(train_iter, optimizer, device=opt.gpu)
    # Trainer
    trainer = training.Trainer(updater, (opt.n_epochs, 'epoch'), opt.save)
    trainer.extend(extensions.ExponentialShift('lr', 0.1, opt.LR),
                   trigger=ManualScheduleTrigger(opt.schedule, 'epoch'))
    trainer.extend(extensions.Evaluator(val_iter, model,
                                        device=opt.gpu), trigger=(1, 'epoch'))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(filename='min_loss'), trigger=MinValueTrigger(
        key='validation/main/loss', trigger=(5, 'epoch')))
    trainer.extend(extensions.snapshot(filename='max_accuracy'), trigger=MaxValueTrigger(
        key='validation/main/accuracy', trigger=(5, 'epoch')))
    trainer.extend(extensions.snapshot_object(model, 'min_loss_model'),
                   trigger=MinValueTrigger(key='validation/main/loss', trigger=(5, 'epoch')))
    trainer.extend(extensions.snapshot_object(model, 'max_accuracy_model'),
                   trigger=MaxValueTrigger(key='validation/main/accuracy', trigger=(5, 'epoch')))
    trainer.extend(extensions.observe_lr())
    trainer.extend(extensions.LogReport())
    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(
            ['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png'))
        trainer.extend(extensions.PlotReport(
            ['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy.png'))
        trainer.extend(extensions.PlotReport(
            ['lr'], 'epoch', file_name='learning_rate.png'))
    trainer.extend(extensions.PrintReport(['elapsed_time', 'epoch', 'iteration', 'lr',
                                           'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar(update_interval=25))
    if opt.resume and os.path.exists(opt.resume):
        chainer.serializers.load_npz(opt.resume, trainer)
    # Run the training
    try:
        trainer.run()
    except Exception as e:
        import shutil
        import traceback
        print('\nerror message')
        print(traceback.format_exc())
        shutil.rmtree(opt.save)


if __name__ == '__main__':
    main()
