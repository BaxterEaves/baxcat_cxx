//
//  template_helpers.hpp
//  baxcat_cxx_distributions
//
//  Created by Baxter Eaves on 6/28/14.
//  Copyright (c) 2014 Baxter Eaves. All rights reserved.
//

#ifndef baxcat_cxx_distributions_template_helpers_hpp
#define baxcat_cxx_distributions_template_helpers_hpp


namespace baxcat{

    template <typename Condition, typename T = void>
    using enable_if = typename std::enable_if<Condition::value, T>::type;

    template <typename Condition, typename T = void>
    using disable_if = typename std::enable_if<!Condition::value, T>::type;
}

#endif
