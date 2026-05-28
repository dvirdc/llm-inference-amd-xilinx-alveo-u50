// arg_parser.hpp -- minimal CLI parser shared by all four apps.
//
// Why a custom parser:
//   * Apps share a lot of flag *names* but differ in which ones are
//     required (main_cpu doesn't need --xclbin; inspect_model only wants
//     --checkpoint). Hard-coding required-args into one parser means each
//     app can register only the flags it uses, in two lines.
//   * Zero deps -- same reason as logging.hpp.
//   * Header-only so each app links its own copy at no cost.
//
// Supported forms:
//   --key value          (string / int / float values)
//   --flag               (toggle -- argless, presence sets it to true)
//   --key=value          (also accepted)
//
// Anything that doesn't parse exits the program with a clear error.

#pragma once

#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace m2 {

class ArgParser {
public:
    explicit ArgParser(std::string program_name) : prog_(std::move(program_name)) {}

    void add_string(const std::string& key, const std::string& help,
                    const std::string& def = "") {
        opts_.push_back({key, help, def, /*toggle*/false});
        values_[key] = def;
    }
    void add_toggle(const std::string& key, const std::string& help) {
        opts_.push_back({key, help, "false", /*toggle*/true});
        values_[key] = "false";
    }

    void parse(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            std::string tok = argv[i];
            if (tok == "-h" || tok == "--help") { print_help(); std::exit(0); }

            std::string key, val;
            auto eq = tok.find('=');
            if (eq != std::string::npos) {
                key = tok.substr(0, eq);
                val = tok.substr(eq + 1);
            } else {
                key = tok;
                val = "";
            }
            if (values_.find(key) == values_.end()) {
                std::fprintf(stderr, "unknown arg: %s\n", key.c_str());
                print_help();
                std::exit(2);
            }
            bool toggle = is_toggle(key);
            if (toggle) {
                values_[key] = "true";
            } else {
                if (val.empty()) {
                    if (i + 1 >= argc) {
                        std::fprintf(stderr, "arg %s requires a value\n", key.c_str());
                        std::exit(2);
                    }
                    val = argv[++i];
                }
                values_[key] = val;
            }
        }
    }

    bool        present(const std::string& key) const { return values_.count(key) > 0; }
    std::string get(const std::string& key) const { return values_.at(key); }
    bool        flag(const std::string& key) const { return values_.at(key) == "true"; }
    int         get_int(const std::string& key) const {
        auto v = get(key);
        return v.empty() ? 0 : std::atoi(v.c_str());
    }
    float       get_float(const std::string& key) const {
        auto v = get(key);
        return v.empty() ? 0.0f : static_cast<float>(std::atof(v.c_str()));
    }
    unsigned    get_uint(const std::string& key) const {
        auto v = get(key);
        return v.empty() ? 0u : static_cast<unsigned>(std::strtoul(v.c_str(), nullptr, 0));
    }

    void require(const std::string& key) const {
        auto v = get(key);
        if (v.empty()) {
            std::fprintf(stderr, "missing required arg: %s\n", key.c_str());
            print_help();
            std::exit(2);
        }
    }
    void print_help() const {
        std::fprintf(stderr, "Usage: %s [options]\n", prog_.c_str());
        for (auto& o : opts_) {
            std::fprintf(stderr, "  %-32s %s%s%s\n",
                         (o.key + (o.toggle ? "" : " <val>")).c_str(),
                         o.help.c_str(),
                         o.def.empty() || o.toggle ? "" : "  [default: ",
                         o.def.empty() || o.toggle ? "" : (o.def + "]").c_str());
        }
    }

private:
    struct Opt {
        std::string key, help, def;
        bool toggle;
    };
    bool is_toggle(const std::string& key) const {
        for (auto& o : opts_) if (o.key == key) return o.toggle;
        return false;
    }
    std::string prog_;
    std::vector<Opt> opts_;
    std::map<std::string, std::string> values_;
};

} // namespace m2
