(define (stream construction)
  (:stream sample-print
    :inputs (?e)
    :domain (Element ?e)
    ; :fluents (Printed)
    :outputs (?t)
    :certified (and (PrintAction ?e ?t)
                    (Traj ?t)
               )
  )
)
