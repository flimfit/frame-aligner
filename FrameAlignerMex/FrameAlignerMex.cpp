#include "MexUtils.h"

#include "AbstractFrameAligner.h"
#include <string>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <limits>

using namespace std;


void Cleanup();

vector<unique_ptr<AbstractFrameAligner>> aligners;

void mexFunction(int nlhs, mxArray *plhs[],
   int nrhs, const mxArray *prhs[])
{
   mexAtExit(Cleanup);
   AssertInputCondition(nrhs >= 1);

   try
   {
      if ((nrhs == 1) && mxIsChar(prhs[0]))
      {
         AssertInputCondition(nlhs == 1);

         string filename = getStringFromMatlab(prhs[0]);

         auto aligner = unique_ptr<AbstractFrameAligner>();

         // Make sure we have an empty place
         if (aligners.empty() || aligners[aligners.size()-1] != nullptr)
            aligners.push_back(unique_ptr<AbstractFrameAligner>(nullptr));

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
         string command = getStringFromMatlab(prhs[1]);

         if (idx >= aligners.size() || aligners[idx] == nullptr)
            mexErrMsgIdAndTxt("FrameAlignerMex:invalidReader",
            "Invalid reader index specified");

         if (command == "GetTimePoints")
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
   catch (exception e)
   {
      mexErrMsgIdAndTxt("FrameAlignerMex:exceptionOccurred",
         e.what());
   }
}


void Cleanup()
{
   aligners.clear();
}
