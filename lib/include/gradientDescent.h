#ifndef GRADIENTDESCENT_H
#define GRADIENTDESCENT_H

#include "learningRule.h"

namespace neuralToy
{
    class GradientDescent : public LearningRule
    {
        public:
            GradientDescent(const Network&, Learning);
            virtual ~GradientDescent();
            double get_learning_rate(void);
            void set_learning_rate(double);
            double get_momentum(void);
            void set_momentum(double);
            void train(const DataSet&);

        protected:

        private:
            double learning_rate_;
            double momentum_;
            double*** updated_weight_;
            void update_weight(void);
    };
}

#endif // GRADIENTDESCENT_H
