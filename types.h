#pragma once
#include <limits>
#include "utils.cu"



/*
 *  >>> SPAS DATA TYPES
 *  a data type to be used with SPAS should implement equals() and get_random()
 *
 */
class MyInt {
public:
  typedef int32_t ElTp;

  static __host__ inline
  bool equals(const int32_t t1, const int32_t t2) { return t1 == t2; }

  static __device__ inline
  int32_t get_random(uint32_t seed) { // random numbers in range -74712 .. 74743.
    const uint32_t m = 149489;        // random enough for our purposes, anyway :) tjktjk
    return (seed * 214741) % m - m/2;
  }
};

class MyFloat {
public:
  typedef float ElTp;

  static __host__ inline
  bool equals(const float t1, const float t2) { return abs(t1 - t2) < 0.1f; }


  static __device__ inline
  float get_random(uint32_t seed) {
    const uint32_t m = 149489;   // random numbers roughly in the range -74 .. 74
    return (float) ((seed * 214741) % m - m/2) / 1000.0f;
  }
};

class MyBool {
public:
  typedef bool ElTp;

  static __host__ inline
  bool equals(const bool b1, const bool b2) { return b1 == b2; }

  static __device__ inline
  bool get_random(uint32_t seed) {
    return ((seed * 214741) % 149489) % 2 == 0;
  }
};

/*
 *  tuple type used for scanning during parallel lookback.
 */
template<class T>
class ValFlg {
public:
  T       v;
  uint8_t f;
  __device__ __host__ inline
  ValFlg() { f = (uint8_t) 0; }

  __device__ __host__ inline
  ValFlg(const uint8_t& f1, const T& v1) { v = v1; f = f1; }

  __device__ __host__ inline
  ValFlg(const ValFlg& vf) { v = vf.v; f = vf.f; }

  __device__ __host__ inline
  void operator=(const ValFlg& vf) volatile { v = vf.v; f = vf.f; }
};



template <class OP>
class MyValFlg {
public:
  uint64_t valflg;

  __device__ inline
  typename OP::ElTp v() volatile {
    return static_cast<typename OP::ElTp>(valflg >> 2);
  }

  __device__ inline
  uint8_t f() volatile {
    return static_cast<uint8_t>(valflg & 3); // extract first two bits
  }

  __device__ __host__ inline
  void operator=(const MyValFlg<OP>& vf) volatile {
    valflg = vf.valflg;
  }

  __device__ inline
  MyValFlg(const volatile MyValFlg<OP> &vf) : valflg(vf.valflg) { }

  __device__ inline
  MyValFlg(const MyValFlg<OP> &vf) : valflg(vf.valflg) { }

  __device__ inline
  MyValFlg(const typename OP::ElTp &v, const uint8_t &f)
   : valflg((static_cast<uint64_t>(v) << 2) | f) { }
};

template <class OP>
class MyValFlgOp {
public:

  typedef MyValFlg<OP> ElTp;

  static __device__ inline
  ElTp apply(volatile ElTp &vf1, volatile ElTp &vf2) {
    return ElTp(vf2.f() ? vf2.v() : OP::apply(vf1.v(), vf2.v()),
                vf1.f() | vf2.f());
  }

  static __device__ __host__ inline
  ElTp remVolatile(volatile ElTp &vf) {
    return ElTp(vf.v(), vf.f());
  }

};

// template <class T>
// class MyValFlg {
// public:
//   typedef typename T::ElTp ElTp;
//
//   uint64_t valflg;
//
//   __device__ inline
//   ElTp v() volatile {
//     return static_cast<ElTp>(valflg >> 2);
//   }
//
//   __device__ inline
//   uint8_t f() volatile {
//     return static_cast<uint8_t>(valflg & 3); // extract first two bits
//   }
//
//   static __device__ inline
//   MyValFlg<T> apply(MyValFlg<T> &vf1, MyValFlg<T> &vf2) {
//     return MyValFlg<T>(T::apply(vf2.f() ? T::identity() : vf1.v(), vf2.v()),
//                        vf1.f() | vf2.f());
//   }
//
//   static __device__ __host__ inline
//   ElTp apply(volatile ElTp &fvp1, volatile ElTp &fvp2) {
//     return ElTp(fvp1.f | fvp2.f,
//                 OP::apply(fvp2.f ? OP::identity() : fvp1.v, fvp2.v));
//   }
//
//   static __device__ __host__ inline
//   MyValFlg<T> remVolatile(volatile MyValFlg<T> &vf) {
//     return MyValFlg(vf.v(), vf.f());
//   }
//
//   __device__ __host__ inline
//   void operator=(const MyValFlg<T>& vf) volatile {
//     valflg = vf.valflg;
//   }
//
//   __device__ inline
//   MyValFlg(const volatile MyValFlg<T> &vf) : valflg(vf.valflg) { }
//
//   __device__ inline
//   MyValFlg(const MyValFlg<T> &vf) : valflg(vf.valflg) { }
//
//   __device__ inline
//   MyValFlg(const ElTp &v, const uint8_t &f)
//    : valflg((uint64_t) ((static_cast<uint64_t>(v) << 2) | f)) { }
// };


/*
 *  >> SCAN OPERATORS
 *
 *  a binary operator to be used with SPAS should inherit class BinOp and must
 *  implement its own "ElTp apply(ElTp, ElTp)" and "ElTp identity()" methods.
 *  
 *  add more where needed.
 */
template <class T>
class BinOp {
public:
  typedef typename T::ElTp ElTp;

  static __host__ inline
  bool equals(const ElTp t1, const ElTp t2) { return T::equals(t1, t2); }

  static __device__ __host__ inline
  ElTp remVolatile(volatile ElTp &t) { ElTp res = t; return res; }

  static __device__ inline
  ElTp get_random() { return T::get_random(); }
};


template<class T>
class Add : public BinOp<T> {
public:
  typedef typename T::ElTp ElTp;

  static __device__ __host__ inline ElTp
  apply(const ElTp t1, const ElTp t2) { return t1 + t2; }

  static __device__ __host__ inline
  ElTp identity() { return (ElTp) 0.0f; }
};


template<class T>
class Mult : public BinOp<T> {
public:
  typedef typename T::ElTp ElTp;

  static __device__ __host__ inline
  ElTp apply(const ElTp t1, const ElTp t2) { return (ElTp) t1 * (ElTp) t2; }

  static __device__ __host__ inline
  ElTp identity() { return (ElTp) 1.0f; }
};



/*
 *  flag/value pair scan operator
 */
template<class OP>
class FVpairOP {
public:
  typedef ValFlg<typename OP::ElTp> ElTp;

  static __device__ __host__ inline
  ElTp identity() { return ElTp(flag_A, OP::identity()); }

  static __device__ __host__ inline
  ElTp apply(volatile ElTp &fvp1, volatile ElTp &fvp2) {
    return ElTp(fvp1.f | fvp2.f,
                OP::apply(fvp2.f ? OP::identity() : fvp1.v, fvp2.v));
  }

  static __device__ __host__ inline
  bool equals(const ElTp fvp1, const ElTp fvp2) {
    return fvp1.f == fvp2.f && OP::equals(fvp1.v, fvp2.v);
  }

  static __device__ __host__ inline
  ElTp remVolatile(volatile ElTp &fvp) {
    ElTp res;
    res.v = fvp.v;
    res.f = fvp.f;
    return res;
  }
};
