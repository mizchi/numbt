default: check test

fmt:
  moon fmt

check:
  moon check --target native

test:
  moon test --target native

info:
  moon info

release-check: fmt info check test
