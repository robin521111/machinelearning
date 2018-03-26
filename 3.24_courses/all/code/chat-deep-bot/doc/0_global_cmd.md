* globalArgs.add_argument('--test',
                                nargs='?',
                                choices=[Chatbot.TestMode.ALL, Chatbot.TestMode.INTERACTIVE, Chatbot.TestMode.DAEMON],
                                const=Chatbot.TestMode.ALL, default=None,
                                help='if present, launch the program try to answer all sentences from data/test/ with'
                                     ' the defined model(s), in interactive mode, the user can wrote his own sentences,'
                                     ' use daemon mode to integrate the chatbot in another program')
清除原有模型
* globalArgs.add_argument('--reset', action='store_true', help='use this if you want to ignore the previous model present on the model directory (Warning: the model will be destroyed with all the folder content)')
日志打印，当测试时会同时打印输出
* globalArgs.add_argument('--verbose', action='store_true', help='When testing, will plot the outputs at the same time they are computed')
保存所有模型
* globalArgs.add_argument('--keepAll', action='store_true', help='If this option is set, all saved model will be kept (Warning: make sure you have enough free disk space or increase saveEvery)')  # TODO: Add an option to delimit the max size
训练或预测指定模型标签
* globalArgs.add_argument('--modelTag', type=str, default=None, help='tag to differentiate which model to store/load')
指定模型和数据的根目录
* globalArgs.add_argument('--rootDir', type=str, default=None, help='folder where to look for the models and data')
选用设备
* globalArgs.add_argument('--device', type=str, default=None, help='\'gpu\' or \'cpu\' (Warning: make sure you have enough free RAM), allow to choose on which hardware run the model')
