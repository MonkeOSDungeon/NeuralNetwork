# NeuralNetwork
YOLO model to detect people for Operating System subject in SSAU

yolov5_zones.py - нейросеть, распознающая людей в указанной зоне, оформленная в класс

yolov5_zones_email.py - программа, которая в случае распознавания людей в зоне отправляет письмо с кадром на указанную почту
Почта получателя указывается в main()
Почта отправителя и пароль от неё указывается в функции log_in()
Тема письма и текст который будет в нем указывается в функции send_email()
