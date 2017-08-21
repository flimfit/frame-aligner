#include "MexUtils.h"

#include "AbstractFrameAligner.h"
#include <string>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <limits>

void Cleanup();

std::vector<std::unique_ptr<AbstractFrameAligner>> aligners;

void mexFunction(int nlhs, mxArray *plhs[],
   int nrhs, const mxArray *prhs[])
{
   mexAtExit(Cleanup);
   AssertInputCondition(nrhs >= 1);

   try
   {
      if (nrhs == 1)
      {
         AssertInputCondition(nlhs == 1);
         AssertInputCondition(mxIsStruct(prhs[0]));

         RealignmentParameters params;
         params.type = (RealignmentType) ((int) getValueFromStruct(prhs[0], "type", 0));
         params.spatial_binning =  getValueFromStruct(prhs[0], "spatial_binning", 0);
         params.frame_binning = getValueFromStruct(prhs[0], "frame_binning", 0);
         params.n_resampling_points = getValueFromStruct(prhs[0], "n_resampling_points", 10);
         params.smoothing = getValueFromStruct(prhs[0], "smoothing", 0);
         params.correlation_threshold = getValueFromStruct(prhs[0], "correlation_threshold", 0);
         params.coverage_threshold = getValueFromStruct(prhs[0], "coverage_threshold", 0);

         auto aligner = AbstractFrameAligner::createFrameAligner(params);

         // Make sure we have an empty place
         if (aligners.empty() || aligners[aligners.size()-1] != nullptr)
            aligners.push_back(nullptr);

         int i = 0;
         for (; i < aligners.size(); i++)
            if (aligners[i] == nullptr)
            {
               aligners[i].swap(aligner);
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

         if (command == "SetReference")
         {
            AssertInputCondition(nlhs >= 1);
            //...
         }
         else if (command == "AddFrame")
         {
            AssertInputCondition(nlhs >= 1);
            //...
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
   catch (std::exception e)
   {
      mexErrMsgIdAndTxt("FrameAlignerMex:exceptionOccurred",
         e.what());
   }
}


void Cleanup()
{
   aligners.clear();
}
