all:
	g++ \
	-I/usr/local/include \
	-L/usr/local/lib \
	-L../3rdparty/SeetaFaceEngine/bin \
	-lopencv_core -lopencv_imgcodecs -lopencv_objdetect -lopencv_imgproc -lopencv_highgui -lopencv_videoio \
	-lopencv_video \
	-lglfw \
	-lseeta_fa_lib \
	main.cpp \
	../src/laptopVR.cpp \
	-o vr
	rsync -rupE ../resource/* ./resource/
	rsync -rupE ../3rdparty/SeetaFaceEngine/resource/* ./resource/
	rsync -rupE ../3rdparty/SeetaFaceEngine/bin/* ./
