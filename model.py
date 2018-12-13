from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple

from module import *

class cyclegan():
    def __init__(self, sess, checkpoint_dir, test_dir, dataset_dir, which_direction):
        self.sess = sess
        self.batch_size = 1           # 배치 사이즈
        self.image_size = 256            # 잘리는 사이즈
        self.input_c_dim = 3            # 인풋 이미지 채널
        self.output_c_dim = 3          # 아웃풋 이미지 채널
        self.L1_lambda = 10.0
        self.fine_size = 256
        self.ngf = 64         # G의 첫번째 conv 레이어 필터수
        self.ndf = 64         # F의 첫번째 conv 레이어 필터수
        self.output_nc = 3
        self.max_size = 50
        self.beta1 = 0.5        # adam 두번째 옵션
        self.epoch = 200        # 200에폭 돌려라.
        self.epoch_step = 100   # lr를 줄이는 스탭
        self.train_size = 1e8   # 훈련시 이미지 갯수??
        self.lr_init = 0.0002        # 최초의 lr
        self.load_size = 286
        self.save_freq = 500    # 저장 빈도
        self.continue_train = True         # 항상 연속해서 훈련.

        # 경로 모음
        self.checkpoint_dir = checkpoint_dir                # checkpoint 경로.
        self.dataset_dir = dataset_dir              # dataset 경로
        self.test_dir = test_dir                    # 테스트한 이미지를 저장하는 경로.

        # G와 D
        self.discriminator = discriminator
        self.generator = generator_resnet  # resnet을 사용한 generator

        # loss_fn
        self.original_GAN_loss = mae_criterion

        self.which_direction = which_direction

        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size \
                                      gf_dim df_dim output_c_dim')
        self.options = OPTIONS._make((self.batch_size, self.fine_size,
                                      self.ngf, self.ndf, self.output_nc,))       #ngf 는 제너레이터 함수의 첫번째 conv레이어 필터수
                                      #self.phase == 'train'))

        self._build_model()         # 모델 생성.
        #writer = tf.train.SummaryWriter("/tmp/test_logs", sess.graph)
        self.saver = tf.train.Saver()
        self.pool = ImagePool(self.max_size)

    def _build_model(self):
        self.real_data = tf.placeholder(tf.float32,
                                        [None, self.image_size, self.image_size,
                                         self.input_c_dim + self.output_c_dim],     # input 이미지 채널과 output 이미지 채널 수  # 아마 둘다 3. RGB라서.
                                        name='real_A_and_B_images')

        self.real_A = self.real_data[:, :, :, :self.input_c_dim]            # self.real_data의 앞부분, 즉 real A
        self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim] # real B
        self.fake_B = self.generator(self.real_A, self.options, False, name="generatorA2B")     # A를 가짜 B로 바꾸기.
        self.fake_A_ = self.generator(self.fake_B, self.options, False, name="generatorB2A")    # 가짜 B를 가짜 A로 바꾸기

        # 앞에 이미 정의한 거라서 reuse를 사용함.
        self.fake_A = self.generator(self.real_B, self.options, True, name="generatorB2A")      # 진짜 B를 가짜 A로 바꾸기
        self.fake_B_ = self.generator(self.fake_A, self.options, True, name="generatorA2B")     # 가짜 A를 가짜 B로 바꾸기
        self.DB_fake = self.discriminator(self.fake_B, self.options, reuse=False, name="discriminatorB")        # 32 BY 32 가 나와..
        self.DA_fake = self.discriminator(self.fake_A, self.options, reuse=False, name="discriminatorA")

        """  G & F 를 학습 시키키 위해 필요한 최종 loss  """
        self.g_loss = self.original_GAN_loss(self.DA_fake, tf.ones_like(self.DA_fake)) \
            + self.original_GAN_loss(self.DB_fake, tf.ones_like(self.DB_fake)) \
            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)

        """    신규 추가 loss   
        #########################################################
        self.A_and_GA_hyunbo = self.generator(self.real_A, self.options, True, name="generatorB2A")
        self.B_and_GB_hyunbo = self.generator(self.real_B, self.options, True, name="generatorA2B")
        self.g_loss_by_hyunbo = self.L1_lambda * abs_criterion(self.real_A, self.A_and_GA_hyunbo) \
                                    + self.L1_lambda * abs_criterion(self.real_B, self.B_and_GB_hyunbo)

        #########################################################"""
        """ =============================== 여기부터는 D의 학습입니다. =========================="""
        # 대문자 D를 함수 처럼 작용하는 듯.
        self.fake_A_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size,
                                             self.input_c_dim], name='fake_A_sample')
        self.fake_B_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size,
                                             self.output_c_dim], name='fake_B_sample')

        # 진짜 B를 넣어서 32 by 32 를 나오게 함.
        self.DB_real = self.discriminator(self.real_B, self.options, reuse=True, name="discriminatorB")
        # 진짜 A를 넣어서 32 by 32 를 나오게 함.
        self.DA_real = self.discriminator(self.real_A, self.options, reuse=True, name="discriminatorA")
        # 가짜 B를 넣어서 32 by 32 를 나오게 함.
        self.DB_fake_sample = self.discriminator(self.fake_B_sample, self.options, reuse=True, name="discriminatorB")
        # 가짜 A를 넣어서 32 by 32 를 나오게 함.
        self.DA_fake_sample = self.discriminator(self.fake_A_sample, self.options, reuse=True, name="discriminatorA")

        """ 진짜 B와 가짜 B를 구별하는 D_loss"""
        self.db_loss_real = self.original_GAN_loss(self.DB_real, tf.ones_like(self.DB_real))
        self.db_loss_fake = self.original_GAN_loss(self.DB_fake_sample, tf.zeros_like(self.DB_fake_sample))
        self.db_loss = (self.db_loss_real + self.db_loss_fake) / 2

        """ 진짜 A와 가짜 A를 구별하는 D_loss """
        self.da_loss_real = self.original_GAN_loss(self.DA_real, tf.ones_like(self.DA_real))
        self.da_loss_fake = self.original_GAN_loss(self.DA_fake_sample, tf.zeros_like(self.DA_fake_sample))
        self.da_loss = (self.da_loss_real + self.da_loss_fake) / 2

        """ D를 학습시키기 위한 최종 loss """
        self.d_loss = self.da_loss + self.db_loss

        """============================================================================================"""

        """ test를 위해 만든 곳 """
        self.test_A = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size,
                                      self.input_c_dim], name='test_A')
        self.test_B = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size,
                                      self.output_c_dim], name='test_B')
        self.testB = self.generator(self.test_A, self.options, True, name="generatorA2B")
        self.testA = self.generator(self.test_B, self.options, True, name="generatorB2A")


        """ D를 학습시킬 때의 변수와 G를 학습시킬 때의 변수를 나눠놓음. """
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        for var in t_vars: print(var.name)

    def train(self):
        """Train cyclegan"""
        self.lr_var = tf.placeholder(tf.float32, None, name='learning_rate')
        self.d_optim = tf.train.AdamOptimizer(self.lr_var, beta1=self.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr_var, beta1=self.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        """
        #########################################################
        self.hyunbo_optim = tf.train.AdamOptimizer(self.lr_var, beta1=self.beta1) \
            .minimize(self.g_loss_by_hyunbo, var_list=self.g_vars)
        #########################################################
        """

        init_op = tf.global_variables_initializer()     #모든 변수 초기화.
        self.sess.run(init_op)

        counter = 1
        start_time = time.time()

        if self.continue_train:
            suc_or_fal, num_of_train = self.load()          # 성공, 실패 유무 와 훈련횟수
            if suc_or_fal:
                print(" check point 가져오기 성공")
                counter = int(num_of_train.split('-')[1])                     # 이어서 저장하기 위해서..
            else:
                print(" check point 가져오기 실패")

        for epoch in range(self.epoch):
            dataA = glob('{}/*.*'.format(self.dataset_dir + '/trainA'))
            dataB = glob('{}/*.*'.format(self.dataset_dir + '/trainB'))
            print("num of dataA : ", dataA.__len__())
            np.random.shuffle(dataA)
            np.random.shuffle(dataB)
            batch_idxs = min(min(len(dataA), len(dataB)), self.train_size) // self.batch_size
            lr = self.lr_init if epoch < self.epoch_step else self.lr_init*(self.epoch-epoch)/(self.epoch-self.epoch_step)    # lr이 작아짐. 100에폭이 지나면.

            for idx in range(0, batch_idxs):
                batch_files = list(zip(dataA[idx * self.batch_size:(idx + 1) * self.batch_size],
                                       dataB[idx * self.batch_size:(idx + 1) * self.batch_size]))
                batch_images = [load_train_data(batch_file, self.load_size, self.fine_size) for batch_file in batch_files]      # 이미지 크기 바꾸기.
                batch_images = np.array(batch_images).astype(np.float32)

                # Update G network and record fake outputs
                fake_A, fake_B, _ = self.sess.run(
                    [self.fake_A, self.fake_B, self.g_optim],
                    feed_dict={self.real_data: batch_images, self.lr_var: lr})

                #self.writer.add_summary(summary_str, counter)
                #[fake_A, fake_B] = self.pool([fake_A, fake_B])

                # Update D network
                self.sess.run( [self.d_optim],
                               feed_dict={self.real_data: batch_images,
                                    self.fake_A_sample: fake_A,
                                    self.fake_B_sample: fake_B,
                                    self.lr_var: lr})
                #self.writer.add_summary(summary_str, counter)

                counter += 1
                print(("Epoch: [%2d] [%4d/%4d] time: %4.4f" % (
                    epoch, idx, batch_idxs, time.time() - start_time)))

                # 저장.
                if np.mod(counter, self.save_freq) == 2:
                    self.save(counter)
                    print("ckpt 저장 완료")

    def save(self, step):

        step_str = str(step)

        self.saver.save(self.sess,
                        self.checkpoint_dir + "/" + step_str,
                        global_step=step)

        """ 여기는 ckpt 파일을 gdrive에 올리는 코드입니다."""
        """latest = tf.train.latest_checkpoint(checkpoint_dir)
        file_name = latest.split("/")
        file_name = file_name[-1]

        list = ['.data-00000-of-00001', '.index', '.meta']
        for x in list:

            drive_service = build('drive', 'v3')
            file_metadata = {
                'name': file_name + x,  # 구글에 올라가는 이름
                'mimeType': None
            }

            media = MediaFileUpload('' + latest + x,
                                    mimetype=None,
                                    resumable=True)
            created = drive_service.files().create(body=file_metadata,
                                                   media_body=media,
                                                   fields='id').execute()
            print('File ID: {}'.format(created.get('id')))
            print(x + "gdrive에 올리기 성공")"""

    def load(self):
        print(" [*] Reading checkpoint...")

        #model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        #checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
            print(" 체크 포인트 이름!! : ", ckpt_name)
            index = ckpt_name.find("-")
            num_of_train = ckpt_name[index + 1:]
            print(" 불러온 ckpt 횟수 : ", num_of_train)
            time.sleep(10)
            return True, num_of_train
        else:
            print("check point가 경로에 없습니다.")
            return False, 1             # 리턴값 맞춰주기 위해 씀. 아무값이나 리턴

    def test(self):
        """Test cyclegan"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        if self.which_direction == 'AtoB':
            sample_files = glob('{}/*.*'.format(self.dataset_dir + '/testA'))
            print("dataset 가져오기 성공.")
        elif self.which_direction == 'BtoA':
            sample_files = glob('{}/*.*'.format(self.dataset_dir + '/testB'))
            print("dataset 가져오기 성공.")
        else:
            raise Exception('AtoB BtoA 둘중 하나는 선택하세요.')

        if self.load():
            print(" check point 가져오기 성공")
        else:
            print(" check point 가져오기 실패")

        if self.which_direction == 'AtoB':
            out_var, in_var = (self.testB, self.test_A)
        else:
            out_var, in_var = (self.testA, self.test_B)

        for sample_file in sample_files:
            print('Processing image: ' + sample_file)
            sample_image = [load_test_data(sample_file, self.fine_size)]
            sample_image = np.array(sample_image).astype(np.float32)
            image_path = os.path.join(self.test_dir,
                                      '{0}_{1}'.format(self.which_direction, os.path.basename(sample_file)))
            fake_img = self.sess.run(out_var, feed_dict={in_var: sample_image})
            print("start saving")
            save_images(fake_img, [1, 1], image_path)


def main():
    checkpoint_dir = './checkpoint/face_256'             # 체크포인트 경로
    test_dir = './test'                          # 테스트 이미지가 저장되는 경로
    dataset_dir = './datasets/face'                         # 데이터셋 위치    trainA trainB testA testB
    phase = "test"  # or test
    which_direction = "BtoA"        # or BtoA .  테스트시 변환 방향
    # 이미지 저장은 test 경로를 따로 만들어서 함.

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)        # gpu를 사용하겠습니다.
    tfconfig.gpu_options.allow_growth = True

    with tf.Session(config=tfconfig) as sess:
        model = cyclegan(sess, checkpoint_dir = checkpoint_dir, test_dir = test_dir,\
                         dataset_dir= dataset_dir, which_direction= which_direction)
        if phase == 'train':
            print(" 훈련 시작")
            model.train()
        elif phase == "test":
            print(" 테스트 시작")
            model.test()
        else:
            print(" train??? test???? 둘중하나는 고르세요.")

if __name__ == '__main__':
    main()
