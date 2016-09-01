all:
	g++ \
	-I/usr/local/include \
	-L/usr/local/lib \
	-lopencv_core -lopencv_imgcodecs -lopencv_objdetect -lopencv_imgproc -lopencv_highgui -lopencv_videoio \
	-lopencv_video \
	-lglfw \
	main.cpp \
	../src/laptopVR.cpp \
	-o vr
	rsync -rupE ../resource/* ./resource/
