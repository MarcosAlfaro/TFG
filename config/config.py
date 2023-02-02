"""
Main config file of video and camera parameters.
"""
import yaml


class ParametersConfig():
    """
    Clase en la que se almacenan los parametros del registration
    """
    def __init__(self, yaml_file='config/parameters.yml'):
        with open(yaml_file) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            print(config)

            self.base_dir = config.get('base_dir')

            self.training_dir = config.get('rutas_imgs').get('training_dir')
            self.validation_dir = config.get('rutas_imgs').get('validation_dir')
            self.centro_geom_dir = config.get('rutas_imgs').get('centro_geom_dir')
            self.test_cloudy_dir = config.get('rutas_imgs').get('test_cloudy_dir')
            self.test_sunny_dir = config.get('rutas_imgs').get('test_sunny_dir')
            self.test_night_dir = config.get('rutas_imgs').get('test_night_dir')

            self.train_csv_dir = config.get('rutas_csv').get('train_csv_dir')
            self.val_csv_dir = config.get('rutas_csv').get('val_csv_dir')
            self.data_csv_dir = config.get('rutas_csv').get('data_csv_dir')
            self.test_csv_dir = config.get('rutas_csv').get('test_csv_dir')
            self.cloudy_csv_dir = config.get('rutas_csv').get('test').get('cloudy_csv_dir')
            self.sunny_csv_dir = config.get('rutas_csv').get('test').get('sunny_csv_dir')
            self.night_csv_dir = config.get('rutas_csv').get('test').get('night_csv_dir')

            self.train_batch_size = config.get('batch_size').get('train_batch_size')
            self.val_batch_size = config.get('batch_size').get('val_batch_size')
            self.test_batch_size = config.get('batch_size').get('test_batch_size')

            self.train_number_epochs = config.get('train_number_epochs')

            self.do_dataparallel = config.get('do_dataparallel')

            self.shuffle_train = config.get('shuffle_data').get('shuffle_train')
            self.shuffle_val = config.get('shuffle_data').get('shuffle_val')
            self.shuffle_test = config.get('shuffle_data').get('shuffle_test')

            self.workers_train = config.get('num_workers').get('workers_train')
            self.workers_val = config.get('num_workers').get('workers_val')
            self.workers_test = config.get('num_workers').get('workers_test')

            self.do_validation = config.get('frequency').get('do_validation')
            self.show_loss = config.get('frequency').get('show_loss')

            self.train_net = config.get('names').get('train_net')
            self.test_net = config.get('names').get('test_net')

            self.hard_pos = config.get('probabilities').get('hard_pos')
            self.hard_neg = config.get('probabilities').get('hard_neg')

            self.same_room = config.get('same_room')

            self.num_iteraciones = config.get('num_iteraciones')

            self.min_val_accuracy = config.get('min_val_accuracy')
            self.accuracy_end_train = config.get('accuracy_end_train')


PARAMETERS = ParametersConfig()
