
#ifndef baxcat_cxx_distributions_template_helpers_hpp
#define baxcat_cxx_distributions_template_helpers_hpp


namespace baxcat{

    template <typename Condition, typename T = void>
    using enable_if = typename std::enable_if<Condition::value, T>::type;

    template <typename Condition, typename T = void>
    using disable_if = typename std::enable_if<!Condition::value, T>::type;
}

#endif
