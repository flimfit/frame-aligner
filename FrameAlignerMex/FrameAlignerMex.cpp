#include "MexUtils.h"

#include "AbstractFrameAligner.h"
#include <string>
#include <memory>

#include "cxxpool.h"

cxxpool::thread_pool pool;

void Cleanup();

std::vector<std::shared_ptr<AbstractFrameAligner>> aligners;

void mexFunction(int nlhs, mxArray *plhs[],
   int nrhs, const mxArray *prhs[])
{
   mexAtExit(Cleanup);
   AssertInputCondition(nrhs >= 1);

   if (pool.n_threads() == 0)
      pool.add_threads(4);

   try
   {
      if ((nrhs == 3) && mxIsStruct(prhs[0]) && mxIsStruct(prhs[1]))
      {
         AssertInputCondition(nlhs == 1);

         RealignmentParameters params;
         params.type = (RealignmentType) ((int) getValueFromStruct(prhs[0], "type", (double) RealignmentType::Warp));
         params.spatial_binning = getValueFromStruct(prhs[0], "spatial_binning", 1);
         params.frame_binning = getValueFromStruct(prhs[0], "frame_binning", 1);
         params.n_resampling_points = getValueFromStruct(prhs[0], "n_resampling_points", 10);
         params.smoothing = getValueFromStruct(prhs[0], "smoothing", 0);
         params.correlation_threshold = getValueFromStruct(prhs[0], "correlation_threshold", 0);
         params.coverage_threshold = getValueFromStruct(prhs[0], "coverage_threshold", 0);

         AssertInputCondition(params.frame_binning >= 1);
         AssertInputCondition(params.spatial_binning >= 1);

         double line_duration = getValueFromStruct(prhs[1], "line_duration", 1);
         double interline_duration = getValueFromStruct(prhs[1], "interline_duration", 1);
         int n_x = getValueFromStruct(prhs[1], "n_x");
         int n_y = getValueFromStruct(prhs[1], "n_y");
         bool bidirectional = getValueFromStruct(prhs[1], "bidirectional", 0);

         ImageScanParameters image_params(line_duration, interline_duration, 1, n_x, n_y, bidirectional);


         int n_frames = mxGetScalar(prhs[2]);

         std::shared_ptr<AbstractFrameAligner> aligner{ AbstractFrameAligner::createFrameAligner(params) };
         
         aligner->setImageScanParams(image_params);
         aligner->setNumberOfFrames(n_frames);

         // Make sure we have an empty place
         if (aligners.empty() || aligners[aligners.size()-1] != nullptr)
            aligners.push_back(nullptr);

         int i = 0;
         for (; i < aligners.size(); i++)
            if (aligners[i] == nullptr)
            {
               aligners[i] = aligner;
               break;
            }
               
         plhs[0] = mxCreateDoubleScalar(i);
      }
      else if (nrhs >= 2)
      {
         if (!mxIsChar(prhs[1]))
            mexErrMsgIdAndTxt("FrameAlignerMex:invalidInput",
            "Second argument should be a command string");

         int idx = static_cast<int>(mxGetScalar(prhs[0]));
         std::string command = getStringFromMatlab(prhs[1]);

         if (idx >= aligners.size() || aligners[idx] == nullptr)
            mexErrMsgIdAndTxt("FrameAlignerMex:invalidReader",
            "Invalid aligner index specified");

         auto aligner = aligners[idx];

         if (command == "SetNumFrames")
         {
            AssertInputCondition(nrhs >= 3);
            int n_frames = mxGetScalar(prhs[2]);
            AssertInputCondition(n_frames > 0);
            aligner->setNumberOfFrames(n_frames);
         }
         if (command == "SetReference")
         {
            AssertInputCondition(nrhs >= 4);
            int frame_t = mxGetScalar(prhs[2]);
            cv::Mat reference = getCvMat(prhs[3]);
            reference.convertTo(reference, CV_32F);
            aligner->setReference(frame_t, reference);
         }
         else if (command == "AddFrame")
         {
            AssertInputCondition(nrhs >= 4);
            int frame_t = mxGetScalar(prhs[2]);
            cv::Mat frame = getCvMat(prhs[3]);
            frame.convertTo(frame, CV_32F);

            // Compute in the thread pool
            pool.push([frame_t, frame, aligner](){
                  aligner->addFrame(frame_t, frame);                  
            });
         }
         else if (command == "NumTasksRemaining")
         {
            if (nlhs == 0) return;
            
            size_t n_tasks = pool.n_tasks();
            plhs[0] = mxCreateDoubleScalar(n_tasks);
         }
         else if (command == "GetRealignedFrame")
         {
            AssertInputCondition(nrhs >= 3);
            AssertInputCondition(nlhs >= 1);
            AssertInputCondition(pool.n_tasks() == 0);

            int frame = mxGetScalar(prhs[2]);
            auto& result = aligner->getResult(frame);

            plhs[0] = convertCvMat(result.realigned);
         }
         else if (command == "Delete")
         {
            aligners[idx] = nullptr;
         }
      }
   }
   catch (std::runtime_error e)
   {
      mexErrMsgIdAndTxt("FrameAlignerMex:runtimeErrorOccurred",
         e.what());
   }
}


void Cleanup()
{
   aligners.clear();
}
