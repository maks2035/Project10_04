#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <mpi.h>

void print_face(cv::Mat im, cv::Mat gauss, cv::CascadeClassifier face_cascade) {
   std::vector<cv::Rect> faces;
   face_cascade.detectMultiScale(gauss, faces, 1.1);
   for (int i = 0; i < faces.size(); i++) {
      cv::rectangle(im, faces[i], cv::Scalar(255, 0, 0), 2);
   }
}

void print_eye(cv::Mat im, cv::Mat gauss, cv::CascadeClassifier eye_cascade) {
   std::vector<cv::Rect> eyes;
   eye_cascade.detectMultiScale(gauss, eyes, 1.1);
   for (int i = 0; i < eyes.size(); i++) {
      cv::Point eye_center(eyes[i].x + eyes[i].width / 2, eyes[i].y + eyes[i].height / 2);
      int radius = cvRound((eyes[i].width + eyes[i].height) * 0.25);
      cv::circle(im, eye_center, radius, cv::Scalar(255, 0, 0), 1);
   }
}

void print_smile(cv::Mat im, cv::Mat gauss, cv::CascadeClassifier smile_cascade) {
   std::vector<cv::Rect> smiles;
   smile_cascade.detectMultiScale(gauss, smiles, 1.565, 30, 0, cv::Size(30, 30));
   for (int i = 0; i < smiles.size(); i++) {
      cv::Point smile_center(smiles[i].x + smiles[i].width / 2, smiles[i].y + smiles[i].height / 2);
      int radius_x = cvRound(smiles[i].width * 0.25);
      int radius_y = cvRound(smiles[i].height * 0.25);
      cv::ellipse(im, smile_center, cv::Size(radius_x, radius_y), 0, 0, 360, cv::Scalar(255, 0, 0), 1);
   }
}

int main(int argc, char** argv) {
   MPI_Init(&argc, &argv);

   int rank, size;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

   cv::VideoCapture cap("D:/virandfpc/vir/Project_09_04/ZUA.mp4");
   if (!cap.isOpened()) {
      std::cout << "Ошибка загрузки первого видео" << std::endl;
      MPI_Finalize();
      return -1;
   }

   cv::CascadeClassifier face_cascade;
   if (!face_cascade.load(cv::samples::findFile("D:/virandfpc/haarcascades/haarcascade_frontalface_alt.xml"))) {
      std::cout << "ERROR" << std::endl;
      MPI_Finalize();
      return -1;
   }

   cv::CascadeClassifier eye_cascade;
   if (!eye_cascade.load(cv::samples::findFile("D:/virandfpc/haarcascades/haarcascade_eye_tree_eyeglasses.xml"))) {
      std::cout << "ERROR" << std::endl;
      MPI_Finalize();
      return -1;
   }

   cv::CascadeClassifier smile_cascade;
   if (!smile_cascade.load(cv::samples::findFile("D:/virandfpc/haarcascades/haarcascade_smile.xml"))) {
      std::cout << "ERROR" << std::endl;
      MPI_Finalize();
      return -1;
   }

   std::vector<cv::Mat> frames;
   int frame_count = 0;
   while (true) {
      cv::Mat frame;
      cap >> frame;
      if (frame.empty()) break;

      if (frame_count % size == rank) { // Определение, какие процессы обрабатывают какие кадры
         cv::resize(frame, frame, cv::Size(), 0.5, 0.5);

         cv::Mat image_gray, gauss;
         cv::cvtColor(frame, image_gray, cv::COLOR_BGR2GRAY);
         cv::GaussianBlur(image_gray, gauss, cv::Size(3, 3), 0);

         print_face(frame, gauss, face_cascade);
         print_eye(frame, gauss, eye_cascade);
         print_smile(frame, gauss, smile_cascade);

         frames.push_back(frame.clone());
      }
      frame_count++;
   }

   std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
   std::chrono::nanoseconds duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

   std::cout << "--------------------------------------------------------------------------" << std::endl;
   std::cout << duration.count() << std::endl;
   std::cout << "--------------------------------------------------------------------------" << std::endl;

   cap.release();

   if (rank == 0) {
      
      cv::VideoWriter video("D:/virandfpc/fpc/Project10_04/output.mp4", cv::VideoWriter::fourcc('a', 'v', 'c', '1'), 30, cv::Size(480, 360));
      if (!video.isOpened()) {
         std::cout << "Error: could not open video writer" << std::endl;
         return -1;
      }

      for (int i = 0; i < frames.size(); i++) {
         cv::imshow("faces detected", frames[i]);
         cv::Mat buff = frames[i].clone();
         cv::resize(buff, buff, cv::Size(480, 360));
         video << buff;

         char c = (char)cv::waitKey(30);
         if (c == 27) break;
      }
      cv::destroyAllWindows();

      video.release();
   }

   MPI_Finalize();

   return 0;
}