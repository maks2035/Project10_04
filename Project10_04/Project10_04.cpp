#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <omp.h>

int main()
{
   std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

   omp_set_num_threads(4);

   cv::VideoCapture cap("D:/virandfpc/vir/Project_09_04/ZUA.mp4");
   if (!cap.isOpened()) {
      std::cout << "Ошибка загрузки первого видео" << std::endl;
      return -1;
   }

   cv::CascadeClassifier face_cascade;
   if (!face_cascade.load(cv::samples::findFile("D:/virandfpc/haarcascades/haarcascade_frontalface_alt.xml"))) {
      std::cout << "ERROR" << std::endl;
      return -1;
   }

   cv::CascadeClassifier eye_cascade;
   if (!eye_cascade.load(cv::samples::findFile("D:/virandfpc/haarcascades/haarcascade_eye_tree_eyeglasses.xml"))) {
      std::cout << "ERROR" << std::endl;
      return -1;
   }

   cv::CascadeClassifier smile_cascade;
   if (!smile_cascade.load(cv::samples::findFile("D:/virandfpc/haarcascades/haarcascade_smile.xml"))) {
      std::cout << "ERROR" << std::endl;
      return -1;
   }

#pragma omp parallel
   {
      while (true) {
         cv::Mat frame;
         cap >> frame;
         if (frame.empty()) break;

         cv::resize(frame, frame, cv::Size(), 0.5, 0.5);

         cv::Mat image_gray, gauss;
         cv::cvtColor(frame, image_gray, cv::COLOR_BGR2GRAY);
         cv::GaussianBlur(image_gray, gauss, cv::Size(3, 3), 0);

         std::vector<cv::Rect> faces;
         std::vector<cv::Rect> eyes;
         std::vector<cv::Rect> smiles;

#pragma omp sections
         {
#pragma omp section
            {
               face_cascade.detectMultiScale(gauss, faces, 1.1);
            }

#pragma omp section
            {
               eye_cascade.detectMultiScale(gauss, eyes, 1.1);
            }

#pragma omp section
            {
               smile_cascade.detectMultiScale(gauss, smiles, 1.565, 30, 0, cv::Size(30, 30));
            }
         }


#pragma omp parallel for
         for (int i = 0; i < faces.size(); ++i) {
            cv::rectangle(frame, faces[i], cv::Scalar(255, 0, 0), 2);
         }

#pragma omp parallel for 
         for (int i = 0; i < eyes.size(); ++i) {
            cv::Point eye_center(eyes[i].x + eyes[i].width / 2, eyes[i].y + eyes[i].height / 2);
            int radius = cvRound((eyes[i].width + eyes[i].height) * 0.25);
            cv::circle(frame, eye_center, radius, cv::Scalar(255, 0, 0), 1);
         }

#pragma omp parallel for
         for (int i = 0; i < smiles.size(); ++i) {
            cv::Point smile_center(smiles[i].x + smiles[i].width / 2, smiles[i].y + smiles[i].height / 2);
            int radius_x = cvRound(smiles[i].width * 0.25);
            int radius_y = cvRound(smiles[i].height * 0.25);
            cv::ellipse(frame, smile_center, cv::Size(radius_x, radius_y), 0, 0, 360, cv::Scalar(255, 0, 0), 1);
         }

#pragma omp critical
         {
            cv::imshow("faces detected", frame);
         }
            char c = (char)cv::waitKey(30);
            if (c == 27) break;
         
      }
   }
   cap.release();
   cv::destroyAllWindows();

   std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
   std::chrono::nanoseconds duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
   std::cout << duration.count() << std::endl;

   return 0;
}

