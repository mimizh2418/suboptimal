#pragma once

template <class F>
  requires std::is_invocable_v<F>
class FinalAction {
public:
  explicit FinalAction(const F& f) : f(f) {}
  explicit FinalAction(F&& f) : f(std::move(f)) {}

  ~FinalAction() { f(); }

  FinalAction(const FinalAction&) = delete;
  void operator=(const FinalAction&) = delete;
  void operator=(FinalAction&&) = delete;
private:
  F f;
};