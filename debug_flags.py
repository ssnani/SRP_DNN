
class ControlFlags(object):

    def __init__(self):
        super().__init__()
        self._TRAIN_ = False
        self._MOVING_SRC_ = True
        self._DOA_OFFLINE_ = "Not Computed" 
        self._TEST_LOCATA_ = True

        self._DEBUG_WAVE_ = False
        self._DEBUG_DOA_ = False

        self.test_exp_name = "test_doA_change_epoch_11" #"train_Single_MovingSrc_lr1e-5_16bit_precision_16k_in_omega"
        self.train_exp_name = "train_Single_MovingSrc_doA_comp_change"
        

    def __repr__(self):

        return f' >>>>> \
                 Exp Name: {self.test_exp_name}, is_train: {self._TRAIN_}, is_moving_src: {self._MOVING_SRC_}, is_doa_offline: {self._DOA_OFFLINE_}, \
                 is_debug_wav: {self._DEBUG_WAVE_}, is_debug_doA: {self._DEBUG_DOA_}, is_test_locata: {self._TEST_LOCATA_ }  \
                   <<<<<<<<<'

    def __str__(self):
        return f' >>>>> \
                 Exp Name: {self.test_exp_name}, is_train: {self._TRAIN_}, is_moving_src: {self._MOVING_SRC_}, is_doa_offline: {self._DOA_OFFLINE_}, \
                 is_debug_wav: {self._DEBUG_WAVE_}, is_debug_doA: {self._DEBUG_DOA_}, is_test_locata: {self._TEST_LOCATA_ }  \
                   <<<<<<<<<'
                   
_control_flags = ControlFlags()