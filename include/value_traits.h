#ifndef VALUE_TRAITS_H
#define VALUE_TRAITS_H

#include <limits.h>
#include <float.h>

template<typename T>
class ValueTraits;

template<>
class ValueTraits<int> {
    public:
        typedef int ValueType;
        static ValueType const MIN = INT_MIN;
        static ValueType const MAX = INT_MAX;
};

template<>
class ValueTraits<unsigned int> {
    public:
        typedef unsigned int ValueType;
        static ValueType const MIN = 0;
        static ValueType const MAX = UINT_MAX;
};

template<>
class ValueTraits<short> {
    public:
        typedef short ValueType;
        static ValueType const MIN = SHRT_MIN;
        static ValueType const MAX = SHRT_MAX;
};

template<>
class ValueTraits<unsigned short> {
    public:
        typedef unsigned short ValueType;
        static ValueType const MIN = 0;
        static ValueType const MAX = USHRT_MAX;
};

template<>
class ValueTraits<long> {
    public:
        typedef long ValueType;
        static ValueType const MIN = LONG_MIN;
        static ValueType const MAX = LONG_MAX;
};

template<>
class ValueTraits<unsigned long> {
    public:
        typedef unsigned long ValueType;
        static ValueType const MIN = 0;
        static ValueType const MAX = ULONG_MAX;
};

template<>
class ValueTraits<float> {
    public:
        typedef float ValueType;
        static ValueType const MIN;
        static ValueType const MAX;
};

template<>
class ValueTraits<double> {
    public:
        typedef double ValueType;
        static ValueType const MIN;
        static ValueType const MAX;
};

template<>
class ValueTraits<long double> {
    public:
        typedef long double ValueType;
        static ValueType const MIN;
        static ValueType const MAX;
};

const float ValueTraits<float>::MIN = FLT_MIN;
const float ValueTraits<float>::MAX = FLT_MAX;
const double ValueTraits<double>::MIN = DBL_MIN;
const double ValueTraits<double>::MAX = DBL_MAX;
const long double ValueTraits<long double>::MIN = LDBL_MIN;
const long double ValueTraits<long double>::MAX = LDBL_MAX;

#endif
