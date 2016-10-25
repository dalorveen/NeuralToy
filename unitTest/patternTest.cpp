#include <gtest/gtest.h>

#include "pattern.h"

TEST(PatternTest, operator_square_brackets)
{
    neuralToy::Pattern pattern(3);
    EXPECT_EQ(0, pattern[0]);
    EXPECT_EQ(0, pattern[1]);
    EXPECT_EQ(0, pattern[2]);
    pattern[0] = 1.2;
    pattern[1] = 3.4;
    pattern[2] = 5.6;
    EXPECT_EQ(1.2, pattern[0]);
    EXPECT_EQ(3.4, pattern[1]);
    EXPECT_EQ(5.6, pattern[2]);
}

TEST(PatternTest, size)
{
    neuralToy::Pattern pattern(4);
    EXPECT_EQ(4, pattern.size());
}

TEST(PatternTest, constructor)
{
    ASSERT_DEATH_IF_SUPPORTED(neuralToy::Pattern pattern_3(0), "size != 0");
}

TEST(PatternTest, operator_equal)
{
    neuralToy::Pattern pattern(3);
    pattern[0] = 1.2;
    pattern[1] = 3.4;
    pattern[2] = 5.6;
    neuralToy::Pattern pattern_2(7);
    pattern_2 = pattern;
    EXPECT_EQ(3, pattern_2.size());
    EXPECT_EQ(1.2, pattern_2[0]);
    EXPECT_EQ(3.4, pattern_2[1]);
    EXPECT_EQ(5.6, pattern_2[2]);
}
