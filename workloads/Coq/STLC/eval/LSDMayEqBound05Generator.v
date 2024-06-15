From QuickChick Require Import QuickChick. Import QcNotation.
From Coq Require Import Bool ZArith List. Import ListNotations.
From ExtLib Require Import Monad.
From ExtLib.Data.Monads Require Import OptionMonad.
Import MonadNotation.

From STLC Require Import Impl Spec.

Inductive CtorTyp :=
  | CtorTyp_TBool
  | CtorTyp_TFun.

Inductive LeafCtorTyp :=
  | LeafCtorTyp_TBool.

Inductive CtorExpr :=
  | CtorExpr_Var
  | CtorExpr_Bool
  | CtorExpr_Abs
  | CtorExpr_App.

Inductive LeafCtorExpr :=
  | LeafCtorExpr_Var
  | LeafCtorExpr_Bool.

Inductive TupCtorTypCtorTyp :=
  | MkCtorTypCtorTyp : CtorTyp -> CtorTyp -> TupCtorTypCtorTyp.

Inductive TupLeafCtorTypLeafCtorTyp :=
  | MkLeafCtorTypLeafCtorTyp : LeafCtorTyp -> LeafCtorTyp -> TupLeafCtorTypLeafCtorTyp.

Inductive TupCtorExprCtorExpr :=
  | MkCtorExprCtorExpr : CtorExpr -> CtorExpr -> TupCtorExprCtorExpr.

Inductive TupCtorTypLeafCtorExpr :=
  | MkCtorTypLeafCtorExpr : CtorTyp -> LeafCtorExpr -> TupCtorTypLeafCtorExpr.

Inductive TupLeafCtorExprLeafCtorExpr :=
  | MkLeafCtorExprLeafCtorExpr : LeafCtorExpr -> LeafCtorExpr -> TupLeafCtorExprLeafCtorExpr.

Inductive TupCtorTypCtorExpr :=
  | MkCtorTypCtorExpr : CtorTyp -> CtorExpr -> TupCtorTypCtorExpr.

Definition genLeafTyp (chosen_ctor : LeafCtorTyp) (stack1 : nat) : G (Typ) :=
  match chosen_ctor with
  | LeafCtorTyp_TBool  => 
    (returnGen (TBool ))
  end.

Fixpoint genTyp (size : nat) (chosen_ctor : CtorTyp) (stack1 : nat) : G (Typ) :=
  match size with
  | O  => match chosen_ctor with
    | CtorTyp_TBool  => 
      (returnGen (TBool ))
    | CtorTyp_TFun  => 
      (bindGen 
      (* Frequency2 (single-branch) *) 
      (returnGen (MkLeafCtorTypLeafCtorTyp (LeafCtorTyp_TBool ) (LeafCtorTyp_TBool ))) 
      (fun param_variantis => (let '(MkLeafCtorTypLeafCtorTyp ctor1 ctor2) := param_variantis in

        (bindGen (genLeafTyp ctor1 1) 
        (fun p1 => 
          (bindGen (genLeafTyp ctor2 7) 
          (fun p2 => 
            (returnGen (TFun p1 p2)))))))))
    end
  | S size1 => match chosen_ctor with
    | CtorTyp_TBool  => 
      (returnGen (TBool ))
    | CtorTyp_TFun  => 
      (bindGen 
      (* Frequency3 *) (freq [
        (* 1 *) (match (size, stack1) with
        | (1, 3) => 93
        | (1, 5) => 90
        | _ => 500
        end,
        (returnGen (MkCtorTypCtorTyp (CtorTyp_TBool ) (CtorTyp_TBool )))); 
        (* 2 *) (match (size, stack1) with
        | (1, 3) => 70
        | (1, 5) => 90
        | _ => 500
        end,
        (returnGen (MkCtorTypCtorTyp (CtorTyp_TFun ) (CtorTyp_TBool )))); 
        (* 3 *) (match (size, stack1) with
        | (1, 3) => 83
        | (1, 5) => 38
        | _ => 500
        end,
        (returnGen (MkCtorTypCtorTyp (CtorTyp_TBool ) (CtorTyp_TFun )))); 
        (* 4 *) (match (size, stack1) with
        | (1, 3) => 86
        | (1, 5) => 93
        | _ => 500
        end,
        (returnGen (MkCtorTypCtorTyp (CtorTyp_TFun ) (CtorTyp_TFun ))))]) 
      (fun param_variantis => (let '(MkCtorTypCtorTyp ctor1 ctor2) := param_variantis in

        (bindGen (genTyp size1 ctor1 2) 
        (fun p1 => 
          (bindGen (genTyp size1 ctor2 8) 
          (fun p2 => 
            (returnGen (TFun p1 p2)))))))))
    end
  end.

Definition genLeafExpr (chosen_ctor : LeafCtorExpr) (stack1 : nat) : G (Expr) :=
  match chosen_ctor with
  | LeafCtorExpr_Var  => 
    (bindGen 
    (* GenNat1 *)
    (let _weight_1 := match (stack1) with
    | (10) => 74
    | (4) => 52
    | (9) => 19
    | _ => 500
    end
    in
    bindGen (freq [
      (_weight_1, returnGen 1);
      (100-_weight_1, returnGen 0)
    ]) (fun n1 =>
    (let _weight_2 := match (stack1) with
    | (10) => 72
    | (4) => 46
    | (9) => 72
    | _ => 500
    end
    in
    bindGen (freq [
      (_weight_2, returnGen 2);
      (100-_weight_2, returnGen 0)
    ]) (fun n2 =>
    (let _weight_4 := match (stack1) with
    | (10) => 80
    | (4) => 63
    | (9) => 27
    | _ => 500
    end
    in
    bindGen (freq [
      (_weight_4, returnGen 4);
      (100-_weight_4, returnGen 0)
    ]) (fun n4 =>
      returnGen (n1 + n2 + n4)
    )))))) 
    (fun p1 => 
      (returnGen (Var p1))))
  | LeafCtorExpr_Bool  => 
    (bindGen 
    (* GenBool1 *) (let _weight_true := match (stack1) with
    | (10) => 20
    | (4) => 50
    | (9) => 76
    | _ => 500
    end
    in
    freq [
      (_weight_true, returnGen true);
      (100-_weight_true, returnGen false)
    ]) 
    (fun p1 => 
      (returnGen (Bool p1))))
  end.

Fixpoint genExpr (size : nat) (chosen_ctor : CtorExpr) (stack1 : nat) : G (Expr) :=
  match size with
  | O  => match chosen_ctor with
    | CtorExpr_Var  => 
      (bindGen 
      (* GenNat2 *)
      (let _weight_1 := match (stack1) with
      | (11) => 47
      | (12) => 43
      | (6) => 49
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_1, returnGen 1);
        (100-_weight_1, returnGen 0)
      ]) (fun n1 =>
      (let _weight_2 := match (stack1) with
      | (11) => 34
      | (12) => 27
      | (6) => 16
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_2, returnGen 2);
        (100-_weight_2, returnGen 0)
      ]) (fun n2 =>
      (let _weight_4 := match (stack1) with
      | (11) => 64
      | (12) => 20
      | (6) => 55
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_4, returnGen 4);
        (100-_weight_4, returnGen 0)
      ]) (fun n4 =>
        returnGen (n1 + n2 + n4)
      )))))) 
      (fun p1 => 
        (returnGen (Var p1))))
    | CtorExpr_Bool  => 
      (bindGen 
      (* GenBool2 *) (let _weight_true := match (stack1) with
      | (11) => 33
      | (12) => 38
      | (6) => 50
      | _ => 500
      end
      in
      freq [
        (_weight_true, returnGen true);
        (100-_weight_true, returnGen false)
      ]) 
      (fun p1 => 
        (returnGen (Bool p1))))
    | CtorExpr_Abs  => 
      (bindGen 
      (* Frequency4 *) (freq [
        (* 1 *) (match (stack1) with
        | (11) => 14
        | (12) => 32
        | (6) => 17
        | _ => 500
        end,
        (returnGen (MkCtorTypLeafCtorExpr (CtorTyp_TBool ) (LeafCtorExpr_Var )))); 
        (* 2 *) (match (stack1) with
        | (11) => 95
        | (12) => 93
        | (6) => 94
        | _ => 500
        end,
        (returnGen (MkCtorTypLeafCtorExpr (CtorTyp_TFun ) (LeafCtorExpr_Var )))); 
        (* 3 *) (match (stack1) with
        | (11) => 8
        | (12) => 20
        | (6) => 8
        | _ => 500
        end,
        (returnGen (MkCtorTypLeafCtorExpr (CtorTyp_TBool ) (LeafCtorExpr_Bool )))); 
        (* 4 *) (match (stack1) with
        | (11) => 8
        | (12) => 67
        | (6) => 20
        | _ => 500
        end,
        (returnGen (MkCtorTypLeafCtorExpr (CtorTyp_TFun ) (LeafCtorExpr_Bool ))))]) 
      (fun param_variantis => (let '(MkCtorTypLeafCtorExpr ctor1 ctor2) := param_variantis in

        (bindGen (genTyp 1 ctor1 3) 
        (fun p1 => 
          (bindGen (genLeafExpr ctor2 9) 
          (fun p2 => 
            (returnGen (Abs p1 p2)))))))))
    | CtorExpr_App  => 
      (bindGen 
      (* Frequency5 *) (freq [
        (* 1 *) (match (stack1) with
        | (11) => 95
        | (12) => 95
        | (6) => 95
        | _ => 500
        end,
        (returnGen (MkLeafCtorExprLeafCtorExpr (LeafCtorExpr_Var ) (LeafCtorExpr_Var )))); 
        (* 2 *) (match (stack1) with
        | (11) => 5
        | (12) => 5
        | (6) => 5
        | _ => 500
        end,
        (returnGen (MkLeafCtorExprLeafCtorExpr (LeafCtorExpr_Bool ) (LeafCtorExpr_Var )))); 
        (* 3 *) (match (stack1) with
        | (11) => 95
        | (12) => 95
        | (6) => 95
        | _ => 500
        end,
        (returnGen (MkLeafCtorExprLeafCtorExpr (LeafCtorExpr_Var ) (LeafCtorExpr_Bool )))); 
        (* 4 *) (match (stack1) with
        | (11) => 5
        | (12) => 5
        | (6) => 5
        | _ => 500
        end,
        (returnGen (MkLeafCtorExprLeafCtorExpr (LeafCtorExpr_Bool ) (LeafCtorExpr_Bool ))))]) 
      (fun param_variantis => (let '(MkLeafCtorExprLeafCtorExpr ctor1 ctor2) := param_variantis in

        (bindGen (genLeafExpr ctor1 4) 
        (fun p1 => 
          (bindGen (genLeafExpr ctor2 10) 
          (fun p2 => 
            (returnGen (App p1 p2)))))))))
    end
  | S size1 => match chosen_ctor with
    | CtorExpr_Var  => 
      (bindGen 
      (* GenNat3 *)
      (let _weight_1 := match (size, stack1) with
      | (1, 11) => 80
      | (1, 12) => 35
      | (1, 6) => 83
      | (2, 11) => 48
      | (2, 12) => 75
      | (2, 6) => 78
      | (3, 11) => 55
      | (3, 12) => 50
      | (3, 6) => 50
      | (4, 0) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_1, returnGen 1);
        (100-_weight_1, returnGen 0)
      ]) (fun n1 =>
      (let _weight_2 := match (size, stack1) with
      | (1, 11) => 76
      | (1, 12) => 35
      | (1, 6) => 44
      | (2, 11) => 26
      | (2, 12) => 14
      | (2, 6) => 60
      | (3, 11) => 42
      | (3, 12) => 50
      | (3, 6) => 50
      | (4, 0) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_2, returnGen 2);
        (100-_weight_2, returnGen 0)
      ]) (fun n2 =>
      (let _weight_4 := match (size, stack1) with
      | (1, 11) => 42
      | (1, 12) => 54
      | (1, 6) => 42
      | (2, 11) => 75
      | (2, 12) => 49
      | (2, 6) => 67
      | (3, 11) => 43
      | (3, 12) => 50
      | (3, 6) => 50
      | (4, 0) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_4, returnGen 4);
        (100-_weight_4, returnGen 0)
      ]) (fun n4 =>
        returnGen (n1 + n2 + n4)
      )))))) 
      (fun p1 => 
        (returnGen (Var p1))))
    | CtorExpr_Bool  => 
      (bindGen 
      (* GenBool3 *) (let _weight_true := match (size, stack1) with
      | (1, 11) => 28
      | (1, 12) => 36
      | (1, 6) => 50
      | (2, 11) => 16
      | (2, 12) => 35
      | (2, 6) => 50
      | (3, 11) => 51
      | (3, 12) => 50
      | (3, 6) => 50
      | (4, 0) => 51
      | _ => 500
      end
      in
      freq [
        (_weight_true, returnGen true);
        (100-_weight_true, returnGen false)
      ]) 
      (fun p1 => 
        (returnGen (Bool p1))))
    | CtorExpr_Abs  => 
      (bindGen 
      (* Frequency6 *) (freq [
        (* 1 *) (match (size, stack1) with
        | (1, 11) => 7
        | (1, 12) => 13
        | (1, 6) => 9
        | (2, 11) => 11
        | (2, 12) => 5
        | (2, 6) => 5
        | (3, 11) => 5
        | (3, 12) => 14
        | (3, 6) => 5
        | (4, 0) => 5
        | _ => 500
        end,
        (returnGen (MkCtorTypCtorExpr (CtorTyp_TBool ) (CtorExpr_Var )))); 
        (* 2 *) (match (size, stack1) with
        | (1, 11) => 13
        | (1, 12) => 18
        | (1, 6) => 20
        | (2, 11) => 20
        | (2, 12) => 6
        | (2, 6) => 5
        | (3, 11) => 5
        | (3, 12) => 26
        | (3, 6) => 6
        | (4, 0) => 5
        | _ => 500
        end,
        (returnGen (MkCtorTypCtorExpr (CtorTyp_TFun ) (CtorExpr_Var )))); 
        (* 3 *) (match (size, stack1) with
        | (1, 11) => 13
        | (1, 12) => 6
        | (1, 6) => 12
        | (2, 11) => 7
        | (2, 12) => 6
        | (2, 6) => 6
        | (3, 11) => 5
        | (3, 12) => 10
        | (3, 6) => 5
        | (4, 0) => 5
        | _ => 500
        end,
        (returnGen (MkCtorTypCtorExpr (CtorTyp_TBool ) (CtorExpr_Bool )))); 
        (* 4 *) (match (size, stack1) with
        | (1, 11) => 6
        | (1, 12) => 11
        | (1, 6) => 8
        | (2, 11) => 10
        | (2, 12) => 5
        | (2, 6) => 10
        | (3, 11) => 5
        | (3, 12) => 23
        | (3, 6) => 5
        | (4, 0) => 5
        | _ => 500
        end,
        (returnGen (MkCtorTypCtorExpr (CtorTyp_TFun ) (CtorExpr_Bool )))); 
        (* 5 *) (match (size, stack1) with
        | (1, 11) => 8
        | (1, 12) => 63
        | (1, 6) => 67
        | (2, 11) => 23
        | (2, 12) => 34
        | (2, 6) => 33
        | (3, 11) => 8
        | (3, 12) => 71
        | (3, 6) => 21
        | (4, 0) => 5
        | _ => 500
        end,
        (returnGen (MkCtorTypCtorExpr (CtorTyp_TBool ) (CtorExpr_Abs )))); 
        (* 6 *) (match (size, stack1) with
        | (1, 11) => 95
        | (1, 12) => 95
        | (1, 6) => 95
        | (2, 11) => 67
        | (2, 12) => 46
        | (2, 6) => 84
        | (3, 11) => 29
        | (3, 12) => 51
        | (3, 6) => 51
        | (4, 0) => 5
        | _ => 500
        end,
        (returnGen (MkCtorTypCtorExpr (CtorTyp_TFun ) (CtorExpr_Abs )))); 
        (* 7 *) (match (size, stack1) with
        | (1, 11) => 26
        | (1, 12) => 5
        | (1, 6) => 10
        | (2, 11) => 83
        | (2, 12) => 91
        | (2, 6) => 56
        | (3, 11) => 88
        | (3, 12) => 78
        | (3, 6) => 85
        | (4, 0) => 95
        | _ => 500
        end,
        (returnGen (MkCtorTypCtorExpr (CtorTyp_TBool ) (CtorExpr_App )))); 
        (* 8 *) (match (size, stack1) with
        | (1, 11) => 71
        | (1, 12) => 27
        | (1, 6) => 18
        | (2, 11) => 86
        | (2, 12) => 95
        | (2, 6) => 95
        | (3, 11) => 95
        | (3, 12) => 79
        | (3, 6) => 91
        | (4, 0) => 95
        | _ => 500
        end,
        (returnGen (MkCtorTypCtorExpr (CtorTyp_TFun ) (CtorExpr_App ))))]) 
      (fun param_variantis => (let '(MkCtorTypCtorExpr ctor1 ctor2) := param_variantis in

        (bindGen (genTyp 1 ctor1 5) 
        (fun p1 => 
          (bindGen (genExpr size1 ctor2 11) 
          (fun p2 => 
            (returnGen (Abs p1 p2)))))))))
    | CtorExpr_App  => 
      (bindGen 
      (* Frequency7 *) (freq [
        (* 1 *) (match (size, stack1) with
        | (1, 11) => 10
        | (1, 12) => 5
        | (1, 6) => 33
        | (2, 11) => 11
        | (2, 12) => 6
        | (2, 6) => 6
        | (3, 11) => 5
        | (3, 12) => 37
        | (3, 6) => 23
        | (4, 0) => 5
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_Var ) (CtorExpr_Var )))); 
        (* 2 *) (match (size, stack1) with
        | (1, 11) => 5
        | (1, 12) => 5
        | (1, 6) => 5
        | (2, 11) => 5
        | (2, 12) => 5
        | (2, 6) => 5
        | (3, 11) => 5
        | (3, 12) => 37
        | (3, 6) => 23
        | (4, 0) => 5
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_Bool ) (CtorExpr_Var )))); 
        (* 3 *) (match (size, stack1) with
        | (1, 11) => 90
        | (1, 12) => 93
        | (1, 6) => 86
        | (2, 11) => 50
        | (2, 12) => 7
        | (2, 6) => 6
        | (3, 11) => 5
        | (3, 12) => 37
        | (3, 6) => 23
        | (4, 0) => 5
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_Abs ) (CtorExpr_Var )))); 
        (* 4 *) (match (size, stack1) with
        | (1, 11) => 65
        | (1, 12) => 81
        | (1, 6) => 49
        | (2, 11) => 51
        | (2, 12) => 5
        | (2, 6) => 29
        | (3, 11) => 5
        | (3, 12) => 37
        | (3, 6) => 23
        | (4, 0) => 5
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_App ) (CtorExpr_Var )))); 
        (* 5 *) (match (size, stack1) with
        | (1, 11) => 12
        | (1, 12) => 25
        | (1, 6) => 17
        | (2, 11) => 6
        | (2, 12) => 5
        | (2, 6) => 5
        | (3, 11) => 5
        | (3, 12) => 37
        | (3, 6) => 23
        | (4, 0) => 5
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_Var ) (CtorExpr_Bool )))); 
        (* 6 *) (match (size, stack1) with
        | (1, 11) => 5
        | (1, 12) => 5
        | (1, 6) => 5
        | (2, 11) => 5
        | (2, 12) => 5
        | (2, 6) => 5
        | (3, 11) => 5
        | (3, 12) => 37
        | (3, 6) => 23
        | (4, 0) => 5
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_Bool ) (CtorExpr_Bool )))); 
        (* 7 *) (match (size, stack1) with
        | (1, 11) => 70
        | (1, 12) => 25
        | (1, 6) => 72
        | (2, 11) => 24
        | (2, 12) => 6
        | (2, 6) => 6
        | (3, 11) => 5
        | (3, 12) => 77
        | (3, 6) => 86
        | (4, 0) => 93
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_Abs ) (CtorExpr_Bool )))); 
        (* 8 *) (match (size, stack1) with
        | (1, 11) => 14
        | (1, 12) => 10
        | (1, 6) => 8
        | (2, 11) => 22
        | (2, 12) => 5
        | (2, 6) => 12
        | (3, 11) => 5
        | (3, 12) => 53
        | (3, 6) => 38
        | (4, 0) => 6
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_App ) (CtorExpr_Bool )))); 
        (* 9 *) (match (size, stack1) with
        | (1, 11) => 93
        | (1, 12) => 95
        | (1, 6) => 95
        | (2, 11) => 16
        | (2, 12) => 8
        | (2, 6) => 14
        | (3, 11) => 5
        | (3, 12) => 37
        | (3, 6) => 23
        | (4, 0) => 5
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_Var ) (CtorExpr_Abs )))); 
        (* 10 *) (match (size, stack1) with
        | (1, 11) => 5
        | (1, 12) => 5
        | (1, 6) => 5
        | (2, 11) => 5
        | (2, 12) => 5
        | (2, 6) => 5
        | (3, 11) => 5
        | (3, 12) => 37
        | (3, 6) => 23
        | (4, 0) => 5
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_Bool ) (CtorExpr_Abs )))); 
        (* 11 *) (match (size, stack1) with
        | (1, 11) => 95
        | (1, 12) => 95
        | (1, 6) => 95
        | (2, 11) => 87
        | (2, 12) => 95
        | (2, 6) => 95
        | (3, 11) => 9
        | (3, 12) => 81
        | (3, 6) => 92
        | (4, 0) => 95
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_Abs ) (CtorExpr_Abs )))); 
        (* 12 *) (match (size, stack1) with
        | (1, 11) => 95
        | (1, 12) => 95
        | (1, 6) => 95
        | (2, 11) => 92
        | (2, 12) => 95
        | (2, 6) => 95
        | (3, 11) => 95
        | (3, 12) => 57
        | (3, 6) => 61
        | (4, 0) => 15
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_App ) (CtorExpr_Abs )))); 
        (* 13 *) (match (size, stack1) with
        | (1, 11) => 79
        | (1, 12) => 30
        | (1, 6) => 22
        | (2, 11) => 37
        | (2, 12) => 12
        | (2, 6) => 25
        | (3, 11) => 6
        | (3, 12) => 37
        | (3, 6) => 23
        | (4, 0) => 5
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_Var ) (CtorExpr_App )))); 
        (* 14 *) (match (size, stack1) with
        | (1, 11) => 5
        | (1, 12) => 5
        | (1, 6) => 5
        | (2, 11) => 5
        | (2, 12) => 5
        | (2, 6) => 5
        | (3, 11) => 5
        | (3, 12) => 37
        | (3, 6) => 23
        | (4, 0) => 5
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_Bool ) (CtorExpr_App )))); 
        (* 15 *) (match (size, stack1) with
        | (1, 11) => 91
        | (1, 12) => 95
        | (1, 6) => 95
        | (2, 11) => 89
        | (2, 12) => 95
        | (2, 6) => 95
        | (3, 11) => 95
        | (3, 12) => 57
        | (3, 6) => 57
        | (4, 0) => 5
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_Abs ) (CtorExpr_App )))); 
        (* 16 *) (match (size, stack1) with
        | (1, 11) => 90
        | (1, 12) => 76
        | (1, 6) => 84
        | (2, 11) => 92
        | (2, 12) => 95
        | (2, 6) => 95
        | (3, 11) => 95
        | (3, 12) => 45
        | (3, 6) => 32
        | (4, 0) => 5
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_App ) (CtorExpr_App ))))]) 
      (fun param_variantis => (let '(MkCtorExprCtorExpr ctor1 ctor2) := param_variantis in

        (bindGen (genExpr size1 ctor1 6) 
        (fun p1 => 
          (bindGen (genExpr size1 ctor2 12) 
          (fun p2 => 
            (returnGen (App p1 p2)))))))))
    end
  end.

Definition gSized :=

  (bindGen 
  (* Frequency1 *) (freq [
    (* 1 *) (match (tt) with
    | tt => 5
    end,
    (returnGen (CtorExpr_Var ))); 
    (* 2 *) (match (tt) with
    | tt => 5
    end,
    (returnGen (CtorExpr_Bool ))); 
    (* 3 *) (match (tt) with
    | tt => 95
    end,
    (returnGen (CtorExpr_Abs ))); 
    (* 4 *) (match (tt) with
    | tt => 5
    end,
    (returnGen (CtorExpr_App )))]) 
  (fun init_ctor => (genExpr 4 init_ctor 0))).

Definition test_prop_SinglePreserve :=
forAll gSized (fun (e: Expr) =>
  prop_SinglePreserve e).

(*! QuickChick test_prop_SinglePreserve. *)

Definition test_prop_MultiPreserve :=
forAll gSized (fun (e: Expr) =>
  prop_MultiPreserve e).

(*! QuickChick test_prop_MultiPreserve. *)
          
