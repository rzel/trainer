import six
import itertools
import numpy as np
from chainer import serializers
from chainer import cuda
import nutszebra_log2
import nutszebra_utility
import nutszebra_sampling
import nutszebra_load_ilsvrc_object_localization
import nutszebra_data_augmentation_picture
import nutszebra_data_augmentation
import nutszebra_basic_print
import multiprocessing

try:
    from cupy.cuda import nccl
    _available = True
except ImportError:
    _available = False

Da = nutszebra_data_augmentation_picture.DataAugmentationPicture()
sampling = nutszebra_sampling.Sampling()
utility = nutszebra_utility.Utility()

"""
I refered the implementation of MultiprocessParallelUpdate for multiprocessing and nccl.
https://github.com/chainer/chainer/blob/master/chainer/training/updaters/multiprocess_parallel_updater.py
"""


class _Worker(multiprocessing.Process):

    def __init__(self, process_id, pipe, model, gpus, da):
        super(_Worker, self).__init__()
        self.process_id = process_id
        self.pipe = pipe
        self.model = model
        self.da = da
        self.device = gpus[process_id]
        self.number_of_devices = len(gpus)
        self.X = []
        self.T = []
        self.Loss = []
        self.Divider = []
        self.Accuracy = []
        self.Accuracy_5 = []
        self.Accuracy_false = []

    def setup(self):
        _, communication_id = self.pipe.recv()
        self.communication = nccl.NcclCommunicator(self.number_of_devices,
                                                   communication_id,
                                                   self.process_id)
        self.model.to_gpu(self.device)

    def run(self):
        dev = cuda.Device(self.device)
        dev.use()
        # build communication via nccl
        self.setup()
        gp = None
        while True:
            job, data = self.pipe.recv()
            if job == 'finalize':
                dev.synchronize()
                break
            if job == 'update':
                # for reducing memory
                self.model.cleargrads()
                x = self.X
                t = self.T
                tmp_x = []
                tmp_t = []
                for i in six.moves.range(len(x)):
                    img, info = self.da.train(x[i])
                    if img is not None:
                        tmp_x.append(img)
                        tmp_t.append(t[i])
                tmp_x = Da.zero_padding(tmp_x)
                train = True
                x = self.model.prepare_input(tmp_x, dtype=np.float32, volatile=not train, gpu=self.device)
                t = self.model.prepare_input(tmp_t, dtype=np.int32, volatile=not train, gpu=self.device)
                y = self.model(x, train=train)
                loss = self.model.calc_loss(y, t) / self.number_of_devices
                loss.backward()
                loss.to_cpu()
                self.Loss.append(float(loss.data * self.number_of_devices * x.data.shape[0]))

                del x
                del t
                del y
                del loss

                # send gradients of self.model
                gg = gather_grads(self.model)
                null_stream = cuda.Stream.null
                self.communication.reduce(gg.data.ptr,
                                          gg.data.ptr,
                                          gg.size,
                                          nccl.NCCL_FLOAT,
                                          nccl.NCCL_SUM,
                                          0,
                                          null_stream.ptr)
                del gg
                self.model.cleargrads()
                # send parameters of self.model
                gp = gather_params(self.model)
                self.communication.bcast(gp.data.ptr,
                                         gp.size,
                                         nccl.NCCL_FLOAT,
                                         0,
                                         null_stream.ptr)
                scatter_params(self.model, gp)
                gp = None
            if job == 'test':
                # for reducing memory
                self.model.cleargrads()
                x = self.X
                t = self.T
                tmp_x = []
                tmp_t = []
                for i in six.moves.range(len(x)):
                    img, info = self.da.test(x[i])
                    if img is not None:
                        tmp_x.append(img)
                        tmp_t.append(t[i])
                tmp_x = Da.zero_padding(tmp_x)
                train = False
                x = self.model.prepare_input(tmp_x, dtype=np.float32, volatile=not train, gpu=self.device)
                t = self.model.prepare_input(tmp_t, dtype=np.int32, volatile=not train, gpu=self.device)
                y = self.model(x, train=train)
                tmp_accuracy, tmp_5_accuracy, tmp_false_accuracy = self.model.accuracy_n(y, t, n=5)
                self.Accuracy.append(tmp_accuracy)
                self.Accuracy_5.append(tmp_5_accuracy)
                self.Accuracy_false.append(tmp_false_accuracy)
                loss = self.model.calc_loss(y, t)
                loss.to_cpu()
                self.Loss.append(float(loss.data * x.data.shape[0]))

                del x
                del t
                del y
                del loss


def size_num_grads(link):
    """Count total size of all gradient arrays of a given link
    Args:
        link (chainer.link.Link): Target link object.
    """
    size = 0
    num = 0
    for param in link.params():
        if param.size == 0:
            continue
        size += param.size
        num += 1
    return size, num


def _batch_memcpy():
    return cuda.cupy.ElementwiseKernel(
        'raw T ptrs, raw X info',
        'raw float32 dst',
        '''
            int id_min = id_pre;
            int id_max = num_src;
            while (id_max - id_min > 1) {
                int id = (id_max + id_min) / 2;
                if (i < info[id]) id_max = id;
                else              id_min = id;
            }
            int id = id_min;
            float *src = (float *)(ptrs[id]);
            int i_dst = i;
            int i_src = i;
            if (id > 0) i_src -= info[id];
            dst[i_dst] = 0;
            if (src != NULL) {
                dst[i_dst] = src[i_src];
            }
            id_pre = id;
        ''',
        'batch_memcpy',
        loop_prep='''
                int num_src = info[0];
                int id_pre = 0;
            ''')


def gather_grads(link):
    """Put together all gradient arrays and make a single array
    Args:
        link (chainer.link.Link): Target link object.
    Return:
        cupy.ndarray
    """
    size, num = size_num_grads(link)

    ptrs = np.empty(num, dtype=np.uint64)
    info = np.empty(num + 1, dtype=np.int32)
    info[0] = 0
    i = 0
    for param in link.params():
        if param.size == 0:
            continue
        ptrs[i] = 0  # NULL pointer
        if param.grad is not None:
            ptrs[i] = param.grad.data.ptr
        info[i + 1] = info[i] + param.size
        i += 1
    info[0] = num

    ptrs = cuda.to_gpu(ptrs, stream=cuda.Stream.null)
    info = cuda.to_gpu(info, stream=cuda.Stream.null)

    return _batch_memcpy()(ptrs, info, size=size)


def gather_params(link):
    """Put together all gradient arrays and make a single array
    Args:
        link (chainer.link.Link): Target link object.
    Return:
        cupy.ndarray
    """
    size, num = size_num_grads(link)

    ptrs = np.empty(num, dtype=np.uint64)
    info = np.empty(num + 1, dtype=np.int32)
    info[0] = 0
    i = 0
    for param in link.params():
        if param.size == 0:
            continue
        ptrs[i] = 0  # NULL pointer
        if param.data is not None:
            ptrs[i] = param.data.data.ptr
        info[i + 1] = info[i] + param.size
        i += 1
    info[0] = num

    ptrs = cuda.to_gpu(ptrs, stream=cuda.Stream.null)
    info = cuda.to_gpu(info, stream=cuda.Stream.null)

    return _batch_memcpy()(ptrs, info, size=size)


def scatter_grads(link, array):
    """Put back contents of the specified array to the related gradient arrays
    Args:
        link (chainer.link.Link): Target link object.
        array (cupy.ndarray): gathered array created by gather_grads()
    """
    offset = 0
    for param in link.params():
        next_offset = offset + param.size
        param.grad = array[offset:next_offset].reshape(param.data.shape)
        offset = next_offset


def scatter_params(link, array):
    """Put back contents of the specified array to the related gradient arrays
    Args:
        link (chainer.link.Link): Target link object.
        array (cupy.ndarray): gathered array created by gather_params()
    """
    offset = 0
    for param in link.params():
        next_offset = offset + param.size
        param.data = array[offset:next_offset].reshape(param.data.shape)
        offset = next_offset


class TrainIlsvrcObjectLocalizationClassificationWithMultiGpus(object):

    def __init__(self, model=None, optimizer=None, load_model=None, load_optimizer=None, load_log=None, load_data=None, da=nutszebra_data_augmentation.DataAugmentationNormalizeBigger, save_path='./', epoch=100, batch=128, gpus=(0, 1, 2, 3), start_epoch=1, train_batch_divide=2, test_batch_divide=2, small_sample_training=None):
        self.model = model
        self.optimizer = optimizer
        self.load_model = load_model
        self.load_optimizer = load_optimizer
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
        self.model_init(model, load_model)
        # create directory
        self.save_path = save_path if save_path[-1] == '/' else save_path + '/'
        utility.make_dir('{}model'.format(self.save_path))
        self._initialized = False
        self._pipes = []
        self._workers = []
        self.communication = None
        self.X = []
        self.T = []
        self.Loss = []
        self.Divider = []
        self.Accuracy = []
        self.Accuracy_5 = []
        self.Accuracy_false = []

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
            log({'model': str(self.model)}, 'model')
        return log

    @staticmethod
    def model_init(model, load_model):
        if load_model is None:
            print('Weight initialization')
            model.weight_initialization()
        else:
            print('loading {}'.format(load_model))
            serializers.load_npz(load_model, model)

    @staticmethod
    def available():
        return _available

    def _send_message(self, message):
        for pipe in self._pipes:
            pipe.send(message)

    def setup_workers(self):
        # work only once
        if self._initialized:
            return
        self._initialized = True

        self.model.cleargrads()
        for i in six.moves.range(1, len(self.gpus)):
            pipe, worker_end = multiprocessing.Pipe()
            worker = _Worker(i, worker_end, self.model, self.gpus, self.da())
            worker.start()
            self._workers.append(worker)
            self._pipes.append(pipe)

        with cuda.Device(self.gpus[0]):
            self.model.to_gpu(self.gpus[0])
            self.da = self.da()
            if len(self.gpus) > 1:
                communication_id = nccl.get_unique_id()
                self._send_message(("set comm_id", communication_id))
                self.communication = nccl.NcclCommunicator(len(self.gpus),
                                                           communication_id,
                                                           0)

    def update_core(self):
        self.setup_workers()
        self._send_message(('update', None))
        with cuda.Device(self.gpus[0]):
            self.model.cleargrads()
            x = self.X
            t = self.T
            tmp_x = []
            tmp_t = []
            for i in six.moves.range(len(x)):
                img, info = self.da.train(x[i])
                if img is not None:
                    tmp_x.append(img)
                    tmp_t.append(t[i])
            tmp_x = Da.zero_padding(tmp_x)
            train = True
            x = self.model.prepare_input(tmp_x, dtype=np.float32, volatile=not train, gpu=self.gpus[0])
            t = self.model.prepare_input(tmp_t, dtype=np.int32, volatile=not train, gpu=self.gpus[0])
            y = self.model(x, train=train)
            loss = self.model.calc_loss(y, t) / len(self.gpus)
            loss.backward()
            loss.to_cpu()
            self.Loss.append(float(loss.data * len(self.gpus) * x.data.shape[0]))

            del x
            del t
            del y
            del loss

            # NCCL: reduce grads
            null_stream = cuda.Stream.null
            if self.communication is not None:
                # send grads
                gg = gather_grads(self.model)
                self.communication.reduce(gg.data.ptr,
                                          gg.data.ptr,
                                          gg.size,
                                          nccl.NCCL_FLOAT,
                                          nccl.NCCL_SUM,
                                          0,
                                          null_stream.ptr)
                # copy grads, gg, to  self.model
                scatter_grads(self.model, gg)
                del gg
            self.optimizer.update()
            if self.communication is not None:
                gp = gather_params(self.model)
                self.communication.bcast(gp.data.ptr,
                                         gp.size,
                                         nccl.NCCL_FLOAT,
                                         0,
                                         null_stream.ptr)

    def test_core(self):
        self.setup_workers()
        self.model.cleargrads()
        self._send_message(('test', None))
        with cuda.Device(self.gpus[0]):
            x = self.X
            t = self.T
            tmp_x = []
            tmp_t = []
            for i in six.moves.range(len(x)):
                img, info = self.da.test(x[i])
                if img is not None:
                    tmp_x.append(img)
                    tmp_t.append(t[i])
            tmp_x = Da.zero_padding(tmp_x)
            train = False
            x = self.model.prepare_input(tmp_x, dtype=np.float32, volatile=not train, gpu=self.gpus[0])
            t = self.model.prepare_input(tmp_t, dtype=np.int32, volatile=not train, gpu=self.gpus[0])
            y = self.model(x, train=train)
            loss = self.model.calc_loss(y, t)
            tmp_accuracy, tmp_5_accuracy, tmp_false_accuracy = self.model.accuracy_n(y, t, n=5)
            self.Accuracy.append(tmp_accuracy)
            self.Accuracy_5.append(tmp_5_accuracy)
            self.Accuracy_false.append(tmp_false_accuracy)
            loss.to_cpu()
            self.Loss.append(float(loss.data * x.data.shape[0]))

            del x
            del t
            del y
            del loss

    def finalize(self):
        self._send_message(('finalize', None))

        for worker in self._workers:
            worker.join()

    def train_one_epoch(self):
        # initialization
        log = self.log
        train_x = self.train_x
        train_y = self.train_y
        batch = self.batch
        gpus = self.gpus
        train_batch_divide = self.train_batch_divide
        batch_of_batch = int(batch / train_batch_divide)
        sum_loss = 0
        yielder = sampling.yield_random_batch_from_category(int(len(train_x) / batch), self.picture_number_at_each_categories, batch, shuffle=True)
        progressbar = utility.create_progressbar(int(len(train_x) / batch), desc='train', stride=1)
        # train start
        for _, indices in six.moves.zip(progressbar, yielder):
            for ii in six.moves.range(0, len(indices), batch_of_batch):
                self.X.clear(), self.T.clear(), self.Loss.clear()
                [(worker.X.clear(), worker.T.clear(), worker.Loss.clear()) for worker in self._workers]
                x = train_x[indices[ii:ii + batch_of_batch]]
                t = train_y[indices[ii:ii + batch_of_batch]]
                n_img = int(float(len(x)) / len(gpus))
                self.X = x[:n_img]
                self.T = t[:n_img]
                for i in six.moves.range(1, len(gpus)):
                    self._workers[i].X = x[i * n_img: (i + 1) * n_img]
                    self._workers[i].T = t[i * n_img: (i + 1) * n_img]
                self.update_core()
                sum_loss = np.sum(self.Loss)
        log({'loss': float(sum_loss)}, 'train_loss')
        print(log.train_loss())

    def test_one_epoch(self):
        # initialization
        log = self.log
        test_x = self.test_x
        test_y = self.test_y
        batch = self.batch
        test_batch_divide = self.test_batch_divide
        batch_of_batch = int(batch / test_batch_divide)
        gpus = self.gpus
        categories = self.categories
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
        self.Accuracy.clear(), self.Accuracy_5.clear(), self.Accuracy_false.clear()
        [worker.Accuracy.clear() for worker in self._workers]
        [worker.Accuracy_5.clear() for worker in self._workers]
        [worker.Accuracy_false.clear() for worker in self._workers]
        for i in progressbar:
            self.X.clear(), self.T.clear(), self.Loss.clear()
            [(worker.X.clear(), worker.T.clear(), worker.Loss.clear()) for worker in self._workers]
            x = test_x[i:i + batch_of_batch]
            t = test_y[i:i + batch_of_batch]
            n_img = int(float(len(x)) / len(gpus))
            self.X = x[:n_img]
            self.T = t[:n_img]
            for i in six.moves.range(1, len(gpus)):
                self._workers[i].X = x[i * n_img: (i + 1) * n_img]
                self._workers[i].T = t[i * n_img: (i + 1) * n_img]
            self.test_core()
        for i in six.moves.range(len(self._workers)):
            self.Accuracy += self._workers[i].Accuracy
            self.Accuracy_5 += self._workers[i].Accuracy_5
            self.Accuracy_false += self._workers[i].Accuracy_false
            self.Loss += self._workers[i].Loss
        for tmp_accuracy in self.Accuracy:
            for key in tmp_accuracy:
                sum_accuracy[key] += tmp_accuracy[key]
        for tmp_5_accuracy in self.Accuracy_5:
            for key in tmp_5_accuracy:
                sum_5_accuracy[key] += tmp_5_accuracy[key]
        for tmp_false_accuracy in self.Accuracy_false:
            for key in tmp_false_accuracy:
                false_accuracy[key] += tmp_false_accuracy[key]
        # sum_loss
        sum_loss += np.sum(self.Loss)
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
        model = self.model
        optimizer = self.optimizer
        epoch = self.epoch
        start_epoch = self.start_epoch
        save_path = self.save_path
        epoch_progressbar = utility.create_progressbar(epoch + 1, desc='epoch', stride=1, start=start_epoch)
        for i in epoch_progressbar:
            self.train_one_epoch()
            # save model
            model.save_model('{}model/{}_{}.model'.format(save_path, model.name, i))
            optimizer(i)
            self.test_one_epoch()
            log.generate_loss_figure('{}loss.jpg'.format(save_path))
            log.generate_accuracy_figure('{}accuracy.jpg'.format(save_path))
            log.save(save_path + 'log.json')
