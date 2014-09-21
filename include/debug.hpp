#ifndef baxcat_cxx_debug_guard
#define baxcat_cxx_debug_guard

    // TODO: refactor some of this redundant code
    /* Custom assert macros.
    The goal of these macros is to provide more information than the standard cassert macros and to
    bypass cython killing casserts.
    */

    #include <cmath>
    #include <iostream>

    #ifdef DEBUG
        // #define PRINT_DEBUG_MESSAGE(os, msg)                                    \
        //     do{                                                                 \
        //         (os) << "DEBUG: " << __FILE__ << "(" << __LINE__ << ") in ";    \
        //         (os) << __PRETTY_FUNCTION__ << std::endl;                       \
        //         (os) << "----" << msg << std::endl                              \
        //     }while(0)

        #define ASSERT( os, test )                                              \
            do{                                                                 \
            if (!(test)){                                                       \
                (os) << "DEBUG: " << __FILE__ << "(" << __LINE__ << ") in ";    \
                (os) << __PRETTY_FUNCTION__ << std::endl;                       \
                (os) << "\t" << "TEST: " << #test << " failed." << std::endl;   \
            }                                                                   \
            }while(0)

        #define ASSERT_INFO( os, msg, test )                                \
            do{                                                             \
            if (!(test)){                                                   \
                (os) << "DEBUG: " << __FILE__ << "(" << __LINE__ << ") in ";\
                (os) << __PRETTY_FUNCTION__ << msg << std::endl;            \
                (os) << "\t" << "TEST: " << #test << std::endl;             \
            }                                                               \
            }while(0)

        #define ASSERT_GREATER_THAN_ZERO(os, number)                                        \
            do{                                                                             \
                if(number <= 0){                                                            \
                    (os) << "DEBUG: " << __FILE__ << "(" << __LINE__ << ") in ";            \
                    (os) << __PRETTY_FUNCTION__ << #number;                                 \
                    (os) << "(" << number << ") should be greater than zero." << std::endl; \
                }                                                                           \
            }while(0)

        #define ASSERT_IS_A_NUMBER(os, number)                                      \
            do{                                                                     \
                if(std::isnan(number) or std::isinf(number)){                       \
                    (os) << "DEBUG: " << __FILE__ << "(" << __LINE__ << ") in ";    \
                    (os) << __PRETTY_FUNCTION__ << #number;                         \
                    (os) << "(" << number << ") is Inf or NaN." << std::endl;       \
                }                                                                   \
            }while(0)

        #define DEBUG_MESSAGE(os, msg)                                                          \
            do{                                                                                 \
                (os) << "DEBUG: " << __FILE__ << "(" << __LINE__ << ") " << msg << std::endl;   \
            }while(0)
    #else
        #define ASSERT( os, test )
        #define ASSERT_INFO( os, msg, test )
        #define ASSERT_GREATER_THAN_ZERO(os, number)
        #define ASSERT_IS_A_NUMBER(os, number)
        #define DEBUG_MESSAGE(os, msg)
    #endif

#endif
