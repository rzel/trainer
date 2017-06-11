import six
import itertools
import numpy as np
from chainer import serializers
import nutszebra_log2
import nutszebra_utility
import nutszebra_sampling
import nutszebra_load_ilsvrc_object_localization
import nutszebra_preprocess_picture
import nutszebra_data_augmentation_picture
import nutszebra_data_augmentation
import nutszebra_basic_print
from multiprocessing import Pool

Da = nutszebra_data_augmentation_picture.DataAugmentationPicture()
sampling = nutszebra_sampling.Sampling()
preprocess = nutszebra_preprocess_picture.PreprocessPicture()
da = nutszebra_data_augmentation_picture.DataAugmentationPicture()
utility = nutszebra_utility.Utility()




def calculate_loss(models, X, T, train, divider=1.0):
    n_img = int(float(len(X)) / len(models))
    args = [(models[i], X[i * n_img: (i + 1) * n_img], T[i * n_img: (i + 1) * n_img], train, divider) for i in six.moves.range(len(models))]

    pool = Pool(len(models))
    return pool.map(wrap_execute, args)


def addgrads(model_teacher, model_students):
    def _addgrads(model_teacher, model_student):
        model_teacher.addgrads(model_student)
        return True

    def wrap_addgrads(arg):
        return _addgrads(*arg)

    args = [(model_teacher, model_student) for model_student in model_students]
    pool = Pool(len(model_student))
    pool.map(wrap_addgrads, args)
    return True


def copyparams(model_teacher, model_students):
    def _copyparams(model_teacher, model_student):
        model_student.copyparams(model_teacher)
        return True

    def wrap_copyparams(arg):
        return _copyparams(*arg)

    args = [(model_teacher, model_student) for model_student in model_students]
    pool = Pool(len(model_students))
    pool.map(wrap_copyparams, args)
    return True


class TrainIlsvrcObjectLocalizationClassificationWithMultiGpus(object):

    def __init__(self, models=None, optimizer=None, load_models=None, load_optimizer=None, load_log=None, load_data=None, da=nutszebra_data_augmentation.DataAugmentationNormalizeBigger, save_path='./', epoch=100, batch=128, gpus=(0, 1, 2, 3), start_epoch=1, train_batch_divide=2, test_batch_divide=2, small_sample_training=None):
        self.models = models
        self.optimizer = optimizer
        self.load_model = load_models
        self.load_optimizers = load_optimizer
        self.load_log = load_log
        self.load_data = load_data
        self.da = da
        self.save_path = save_path
        self.epoch = epoch
        self.batch = batch
        self.gpus = gpus
        self.start_epoch = start_epoch
        self.train_batch_divide = train_batch_divide
        self.test_batch_divide = test_batch_divide
        self.small_sample_training = small_sample_training
        # Generate dataset
        self.train_x, self.train_y, self.test_x, self.test_y, self.picture_number_at_each_categories, self.categories = self.data_init()
        # Log module
        self.log = self.log_init()
        # initializing
        [self.model_init(model, load_model, gpu)for model, load_model, gpu in six.moves.zip(models, load_models, gpus)]
        # create directory
        self.save_path = save_path if save_path[-1] == '/' else save_path + '/'
        utility.make_dir('{}model'.format(self.save_path))

    def data_init(self):
        data = nutszebra_load_ilsvrc_object_localization.LoadDataset(self.load_data)
        train_x, train_y, test_x, test_y = [], [], [], []
        keys = sorted(list(data['val'].keys()))
        picture_number_at_each_categories = []
        for i, key in enumerate(keys):
            if self.small_sample_training is None:
                picture_number_at_each_categories.append(len(data['train_cls'][key]))
                train_x += data['train_cls'][key]
                train_y += [i for _ in six.moves.range(len(data['train_cls'][key]))]
                test_x += data['val'][key]
                test_y += [i for _ in six.moves.range(len(data['val'][key]))]
            else:
                data['train_cls'][key] = sorted(data['train_cls'][key])
                tmp_x = data['train_cls'][key][:int(self.small_sample_training)]
                picture_number_at_each_categories.append(len(tmp_x))
                train_x += tmp_x
                train_y += [i for _ in six.moves.range(len(tmp_x))]
                test_x += data['val'][key]
                test_y += [i for _ in six.moves.range(len(data['val'][key]))]
        categories = keys
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        test_x = np.array(test_x)
        test_y = np.array(test_y)
        return (train_x, train_y, test_x, test_y, picture_number_at_each_categories, categories)

    def log_init(self):
        load_log = self.load_log
        log = nutszebra_log2.Log2()
        if load_log is not None:
            log.load(load_log)
        else:
            log({'are': self.categories}, 'categories')
            log({'parameter': len(self.train_x)}, 'train_parameter')
            log({'parameter': len(self.test_x)}, 'test_parameter')
            for i in six.moves.range(len(self.categories)):
                log({'parameter': float((np.array(self.test_y) == i).sum())}, 'test_parameter_{}'.format(i))
            log({'model': str(self.models)}, 'model')
        return log

    @staticmethod
    def model_init(model, load_model, gpu):
        if load_model is None:
            print('Weight initialization')
            model.weight_initialization()
        else:
            print('loading {}'.format(load_model))
            serializers.load_npz(load_model, model)
        model.check_gpu(gpu)

    def _execute(self, model, x, t, train, divider):
        x = model.prepare_input(x, dtype=np.float32, volatile=not train, gpu=model._device_id)
        t = model.prepare_input(t, dtype=np.int32, volatile=not train, gpu=model._device_id)
        y = model(x, train=train)
        loss = model.calc_loss(y, t) / divider
        if train is True:
            loss.backward()
        loss.to_cpu()
        return float(loss.data)

    def wrap_execute(self, arg):
        return self._execute(*arg)

    def execute(self, arg, n): 
        p = Pool(n)
        return pool.map(self.wrap_execute, args)

    def train_one_epoch(self):
        # initialization
        log = self.log
        models = self.models
        optimizer = self.optimizer
        train_x = self.train_x
        train_y = self.train_y
        batch = self.batch
        train_batch_divide = self.train_batch_divide
        batch_of_batch = int(batch / train_batch_divide)
        sum_loss = 0
        yielder = sampling.yield_random_batch_from_category(int(len(train_x) / batch), self.picture_number_at_each_categories, batch, shuffle=True)
        progressbar = utility.create_progressbar(int(len(train_x) / batch), desc='train', stride=1)
        n_parallel = len(models)
        # train start
        for _, indices in six.moves.zip(progressbar, yielder):
            [model.cleargrads() for model in models]
            for ii in six.moves.range(0, len(indices), batch_of_batch):
                x = train_x[indices[ii:ii + batch_of_batch]]
                t = train_y[indices[ii:ii + batch_of_batch]]
                data_length = len(x)
                tmp_x = []
                tmp_t = []
                for i in six.moves.range(len(x)):
                    img, info = self.da.train(x[i])
                    if img is not None:
                        tmp_x.append(img)
                        tmp_t.append(t[i])

                tmp_x = Da.zero_padding(tmp_x)
                # calculate loss and accuracy
                n_img = int(float(len(tmp_x)) / len(models))
                args = [(models[i], tmp_x[i * n_img: (i + 1) * n_img], tmp_t[i * n_img: (i + 1) * n_img], True, n_parallel * train_batch_divide) for i in six.moves.range(len(models))]
                losses = self.execute(args, len(models)) 
                # results = calculate_loss(models, tmp_x, tmp_t, True, n_parallel * train_batch_divide)
                loss = np.sum([float(r.result()) for r in results])
                # accumulate grads
                addgrads(models[0], models[1:])
                sum_loss += loss * data_length
            optimizer.update()
            # Synchronized update
            copyparams(models[0], models[1:])
        log({'loss': float(sum_loss)}, 'train_loss')
        print(log.train_loss())

    def test_one_epoch(self):
        # initialization
        log = self.log
        models = self.models
        test_x = self.test_x
        test_y = self.test_y
        batch = self.batch
        save_path = self.save_path
        test_batch_divide = self.test_batch_divide
        batch_of_batch = int(batch / test_batch_divide)
        categories = self.categories
        n_parallel = len(models)
        sum_loss = 0
        sum_accuracy = {}
        sum_5_accuracy = {}
        false_accuracy = {}
        for ii in six.moves.range(len(categories)):
            sum_accuracy[ii] = 0
            sum_5_accuracy[ii] = 0
        elements = six.moves.range(len(categories))
        for ii, iii in itertools.product(elements, elements):
            false_accuracy[(ii, iii)] = 0
        progressbar = utility.create_progressbar(len(test_x), desc='test', stride=batch_of_batch)
        for i in progressbar:
            x = test_x[i:i + batch_of_batch]
            t = test_y[i:i + batch_of_batch]
            tmp_x = []
            tmp_t = []
            for i in six.moves.range(len(x)):
                img, info = self.da.test(x[i])
                if img is not None:
                    tmp_x.append(img)
                    tmp_t.append(t[i])
            data_length = len(tmp_x)
            tmp_x = Da.zero_padding(tmp_x)
            # x = models[0].prepare_input(tmp_x, dtype=np.float32, volatile=True)
            # n_img = int(float(x.data.shape[0]) / n_parallel)
            # # parallely calculate loss
            # losses_and_accuracy = Parallel(n_jobs=n_parallel)(delayed(calculate_loss_and_accuracy)(models[i], x[i * n_img: (i + 1) * n_img], t[i * n_img: (i + 1) * n_img], False, test_batch_divide * len(models)) for i in six.moves.range(len(models)))
            # losses, accuracies = list(zip(*losses_and_accuracy))
            # # to_cpu
            # [loss.to_cpu() for loss in losses]
            # sum_loss += np.sum([loss.data for loss in losses]) * data_length
            for accuracy in accuracies:
                tmp_accuracy, tmp_5_accuracy, tmp_false_accuracy = accuracy
                for key in tmp_accuracy:
                    sum_accuracy[key] += tmp_accuracy[key]
                for key in tmp_5_accuracy:
                    sum_5_accuracy[key] += tmp_5_accuracy[key]
                for key in tmp_false_accuracy:
                    false_accuracy[key] += tmp_false_accuracy[key]
                models[0].save_computational_graph(loss, path=save_path)
            del x
            del t
            del losses_and_accuracy
            del losses
            del accuracies
        # sum_loss
        log({'loss': float(sum_loss)}, 'test_loss')
        # sum_accuracy
        num = 0
        for key in sum_accuracy:
            value = sum_accuracy[key]
            log({'accuracy': int(value)}, 'test_accuracy_{}'.format(key))
            num += value
        log({'accuracy': int(num)}, 'test_accuracy')
        # sum_5_accuracy
        num = 0
        for key in sum_5_accuracy:
            value = sum_5_accuracy[key]
            log({'accuracy': int(value)}, 'test_5_accuracy_{}'.format(key))
            num += value
        log({'accuracy': int(num)}, 'test_5_accuracy')
        # false_accuracy
        for key in false_accuracy:
            if key[0] == key[1]:
                pass
            else:
                value = false_accuracy[key]
                log({'accuracy': int(value)}, 'test_accuracy_{}_{}'.format(key[0], key[1]))
        # show logs
        sen = [log.test_loss(), log.test_accuracy(max_flag=True), log.test_5_accuracy(max_flag=True)]
        print('\n'.join(sen))

    def run(self):
        log = self.log
        models = self.models
        optimizer = self.optimizer
        epoch = self.epoch
        start_epoch = self.start_epoch
        save_path = self.save_path
        epoch_progressbar = utility.create_progressbar(epoch + 1, desc='epoch', stride=1, start=start_epoch)
        for i in epoch_progressbar:
            self.train_one_epoch()
            # save graph once
            # save model
            models[0].save_model('{}model/{}_{}.model'.format(save_path, models[0].name, i))
            optimizer(i)
            self.test_one_epoch()
            log.generate_loss_figure('{}loss.jpg'.format(save_path))
            log.generate_accuracy_figure('{}accuracy.jpg'.format(save_path))
            log.save(save_path + 'log.json')
