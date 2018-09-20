subroutine ls_w_w(a, b, p, t, d, e, f, nt, nf)
  integer, intent(in) :: nt, nf
  double precision, intent(in) :: t(nt), d(nt), e(nt)
  double precision, intent(in) :: f(nf)
  double precision, intent(out) :: a(nf),b(nf),p(nf)
  double precision :: div, sft, cft, s, c, ss, cc, sc
  integer :: i,j

  do i = 1, nf

      div = 0d0
      s = 0d0
      c = 0d0
      ss = 0d0
      cc = 0d0
      sc = 0d0

      do j = 1, nt

         sft = sin(f(i)*t(j))
         cft = cos(f(i)*t(j))

         s  = s + d(j)*e(j)*sft
         c  = c + d(j)*e(j)*cft

         ss = ss + sft*sft*e(j)
         cc = cc + cft*cft*e(j)

         sc = sc + sft*cft*e(j)

      end do

      div  = ss*cc-sc*sc
      a(i) = (s*cc - c*sc)/div
      b(i) = (c*ss - s*sc)/div
      p(i) = a(i)*a(i) + b(i)*b(i)

  end do

end subroutine ls_w_w


subroutine ls_wo_w(a, b, p, t, d, f, nt, nf)
  integer, intent(in) :: nt, nf
  double precision, intent(in) :: t(nt), d(nt)
  double precision, intent(in) :: f(nf)
  double precision, intent(out) :: a(nf),b(nf),p(nf)
  double precision :: div, sft, cft, s, c, ss, cc, sc
  integer :: i,j

  do i = 1, nf

      div = 0d0
      s = 0d0
      c = 0d0
      ss = 0d0
      cc = 0d0
      sc = 0d0

      do j = 1, nt

         sft = sin(f(i)*t(j))
         cft = cos(f(i)*t(j))

         s  = s + d(j)*sft
         c  = c + d(j)*cft

         ss = ss + sft*sft
         cc = cc + cft*cft

         sc = sc + sft*cft

      end do

      div  = ss*cc-sc*sc
      a(i) = (s*cc - c*sc)/div
      b(i) = (c*ss - s*sc)/div
      p(i) = a(i)*a(i) + b(i)*b(i)

  end do

end subroutine ls_wo_w

subroutine energy(y,ell,emm,inc)
  double precision, intent(out) :: y
  integer, intent(in) :: ell, emm
  double precision, intent(in) :: inc

  if (ell == 0) then
     y = 1d0

  else if (ell == 1) then
     if (emm == 0) then
        y = cos(inc)**2
     else if (emm == 1) then
        y = 0.5d0*sin(inc)**2
     end if

  else if (ell == 2) then
     if (emm == 0) then
        y = (3d0*cos(inc)**2-1d0)**2/4d0
     else if (emm == 1) then
        y = sin(2d0*inc)**2*3d0/8d0
     else if (emm == 2) then
        y = sin(inc)**4*3d0/8d0
     end if

  else if (ell == 3) then
     if (emm == 0) then
        y = (5d0*cos(3d0*inc)+3d0*cos(inc))**2/64d0
     else if (emm == 1) then
        y = (5d0*cos(2d0*inc)+3d0)**2*sin(inc)**2*3d0/64d0
     else if (emm == 2) then
        y = cos(inc)**2*sin(inc)**4*15d0/8d0
     else if (emm == 3) then
        y = sin(inc)**6*5d0/16d0
     end if

  end if

end subroutine energy

subroutine harvey(y, x, xpo, freq, amp, nx)

  integer, intent(in) :: nx

  double precision, intent(in) :: x(nx)

  double precision, intent(out) :: y(nx)

  double precision, intent(in) :: amp, freq, xpo

  y = amp/freq/(1d0 + (x / freq)**xpo)

end subroutine harvey

subroutine gaussian(y,x,a,b,c,nx)

  integer, intent(in) :: nx

  double precision, intent(in) :: x(nx)

  double precision, intent(out) :: y(nx)

  double precision, intent(in) :: a, b, c

  y = a * exp(-(x - b)*(x - b) / (2d0 * c*c))

end subroutine gaussian

subroutine lnlikelihood(y,e,x,nx)
  implicit none

  integer, intent(in) :: nx
  double precision, intent(in) :: e(nx)
  double precision, intent(in) :: x(nx)
  double precision, intent(out) :: y

  y = sum(-log(e)-x/e)

end subroutine lnlikelihood

subroutine spectral_model(y, x, &
     ells, emms, freqs, amps, widths, &
     incs, hpowers, htaus, hamps, noise, &
     nx, nfreqs, nbacks)
  implicit none

  integer, intent(in) :: nx, nfreqs, nbacks
  double precision, intent(in) :: x(nx)
  double precision, intent(out) :: y(nx)

  integer, intent(in) :: ells(nfreqs), emms(nfreqs)
  double precision, intent(in), dimension(nfreqs) :: &
       freqs, amps, widths, incs
  double precision, intent(in), dimension(nbacks) :: &
       hamps, htaus, hpowers

  double precision, intent(in) :: noise

  integer :: i
  double precision :: nrg

  y = 0d0
  nrg = 0d0

  do i = 1, nfreqs

     call energy(nrg,ells(i),emms(i),incs(i))

     y = y + amps(i)*nrg/(widths(i)+4d0/widths(i)*(x-freqs(i))**2)

  end do

  do i = 1, nbacks

     y = y + hamps(i)/htaus(i)/(1d0 + (x / htaus(i))**hpowers(i))
  end do

  y = y + noise

end subroutine spectral_model



