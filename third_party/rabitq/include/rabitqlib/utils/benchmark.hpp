#pragma once

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "rabitqlib/defines.hpp"

namespace rabitqlib::benchmark {

inline std::vector<size_t> geometric_nprobe_schedule(
    size_t nlist, size_t start = 8, double multiplier = 1.5
) {
    if (nlist == 0 || start == 0 || !std::isfinite(multiplier) || multiplier <= 1.0) {
        throw std::invalid_argument(
            "nlist/start must be positive and nprobe multiplier must be finite and > 1"
        );
    }

    std::vector<size_t> schedule;
    size_t current = std::min(start, nlist);
    schedule.push_back(current);
    while (current < nlist) {
        const long double scaled =
            std::ceil(static_cast<long double>(current) * multiplier);
        size_t next;
        if (scaled >= static_cast<long double>(nlist)) {
            next = nlist;
        } else {
            next = static_cast<size_t>(scaled);
            if (next <= current) {
                next = current + 1;
            }
        }
        schedule.push_back(next);
        current = next;
    }
    return schedule;
}

inline std::vector<size_t> parse_nprobe_list(std::string_view input, size_t nlist) {
    if (nlist == 0 || input.empty()) {
        throw std::invalid_argument("nlist and explicit nprobe list must be non-empty");
    }
    std::vector<size_t> values;
    size_t first = 0;
    while (first <= input.size()) {
        const size_t comma = input.find(',', first);
        const size_t last = comma == std::string_view::npos ? input.size() : comma;
        const std::string_view token = input.substr(first, last - first);
        if (token.empty()) {
            throw std::invalid_argument("empty value in explicit nprobe list");
        }
        size_t value = 0;
        const auto [end, error] =
            std::from_chars(token.data(), token.data() + token.size(), value);
        if (error != std::errc{} || end != token.data() + token.size()) {
            throw std::invalid_argument("invalid nprobe value: " + std::string(token));
        }
        if (value == 0 || value > nlist) {
            throw std::invalid_argument(
                "nprobe value " + std::to_string(value) + " is outside [1, nlist]"
            );
        }
        if (!values.empty() && value <= values.back()) {
            throw std::invalid_argument(
                "nprobe value " + std::to_string(value) + " is not strictly increasing"
            );
        }
        values.push_back(value);
        if (comma == std::string_view::npos) {
            break;
        }
        first = comma + 1;
    }
    return values;
}

inline std::vector<size_t> explicit_nprobe_schedule(
    std::string_view input, size_t nlist, bool append_nlist
) {
    auto schedule = parse_nprobe_list(input, nlist);
    if (append_nlist && schedule.back() != nlist) {
        schedule.push_back(nlist);
    }
    return schedule;
}

inline bool should_stop_after_target(
    bool exact_schedule, double recall, double target_recall
) {
    return !exact_schedule && recall >= target_recall;
}

inline double recall_at_k(
    const PID* results,
    size_t result_columns,
    const PID* groundtruth,
    size_t groundtruth_columns,
    size_t num_queries,
    size_t topk
) {
    if (num_queries == 0 || topk == 0 || result_columns < topk ||
        groundtruth_columns < topk || results == nullptr || groundtruth == nullptr) {
        throw std::invalid_argument("invalid Recall@k pointers or matrix shape");
    }

    size_t matches = 0;
    for (size_t query = 0; query < num_queries; ++query) {
        const PID* result_row = results + query * result_columns;
        const PID* truth_row = groundtruth + query * groundtruth_columns;
        for (size_t result_index = 0; result_index < topk; ++result_index) {
            bool duplicate = false;
            for (size_t previous = 0; previous < result_index; ++previous) {
                if (result_row[previous] == result_row[result_index]) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) {
                continue;
            }
            for (size_t truth_index = 0; truth_index < topk; ++truth_index) {
                if (result_row[result_index] == truth_row[truth_index]) {
                    ++matches;
                    break;
                }
            }
        }
    }
    return static_cast<double>(matches) /
           static_cast<double>(num_queries * topk);
}

inline double median(std::vector<double> values) {
    if (values.empty()) {
        throw std::invalid_argument("median requires at least one value");
    }
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    if (values.size() % 2 != 0) {
        return values[middle];
    }
    return (values[middle - 1] + values[middle]) / 2.0;
}

inline std::string json_escape(std::string_view input) {
    std::ostringstream output;
    for (const unsigned char character : input) {
        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<unsigned>(character) << std::dec;
                } else {
                    output << static_cast<char>(character);
                }
        }
    }
    return output.str();
}

}  // namespace rabitqlib::benchmark
